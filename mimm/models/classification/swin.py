import math
from functools import partial
from typing import Any, Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mimm.layers.adaptive_average_pooling import AdaptiveAveragePool2D

from mimm.models.shared_layers import MLP
from mimm.models.utils import get_pytorch_weights, load_pytorch_weights
from mimm.ops.stochastic_depth import StochasticDepth
from mimm.models._registry import register_model


def _patch_merging_pad(x: mx.array) -> mx.array:
    H, W, _ = x.shape[-3:]
    x = mx.pad(x, [(0, 0), (0, W % 2), (0, H % 2), (0, 0)])
    x0 = x[..., 0::2, 0::2, :]  # ... C, H/2, W/2
    x1 = x[..., 1::2, 0::2, :]  # ... C, H/2, W/2
    x2 = x[..., 0::2, 1::2, :]  # ... C, H/2, W/2
    x3 = x[..., 1::2, 1::2, :]  # ... C, H/2, W/2
    x = mx.concatenate([x0, x1, x2, x3], -1)  # ... H/2, W/2, 4*C
    return x


def _get_relative_position_bias(
    relative_position_bias_table: mx.array,
    relative_position_index: mx.array,
    window_size: List[int],
) -> mx.array:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.reshape(N, N, -1)
    relative_position_bias = relative_position_bias.transpose(2, 0, 1)[None]
    return relative_position_bias


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def __call__(self, x: mx.array):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [...,  H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2, W/2, 2*C
        return x


class PatchMergingV2(nn.Module):
    """Patch Merging Layer for Swin Transformer V2.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)  # difference

    def __call__(self, x: mx.array):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)
        return x


def roll(x: mx.array, shifts: List[int], axes: List[int]) -> mx.array:
    """
    Roll the tensor along the given dimensions.
    Args:
        x (Tensor): input tensor.
        shifts (List[int]): The number of places by which elements are shifted.
        axes (List[int]): The axis along which the elements are shifted.
    Returns:
        Tensor: The rolled tensor.
    """
    output = x
    for shift, axis in zip(shifts, axes):
        if shift == 0:
            continue
        if shift < 0:
            shift = x.shape[axis] + shift
        shift = shift % x.shape[axis]
        # split and concatenate
        splits = mx.split(output, [shift], axis=axis)
        output = mx.concatenate(splits[::-1], axis=axis)
    return output


def normalize(x: mx.array, axis: int, eps: float = 1e-5) -> mx.array:
    """
    Normalize the input tensor along the given axis.
    Args:
        x (Tensor): input tensor.
        axis (int): The axis along which the input tensor is normalized.
        eps (float): A small value to avoid division by zero. Default: 1e-5.
    Returns:
        Tensor: The normalized tensor.
    """
    return x / (mx.linalg.norm(x, 2, axis, keepdims=True) + eps)


def broadcast_arrays(*args):
    shapes = [arg.shape for arg in args]
    ndim = max(len(shape) for shape in shapes)

    # Determine the shape of the broadcasted arrays
    broadcast_shape = [1] * ndim
    for shape in shapes:
        for i, dim in enumerate(shape):
            broadcast_shape[i] = max(broadcast_shape[i], dim)
    # Broadcast each array to the broadcast shape
    broadcasted_arrays = []
    for shape, arg in zip(shapes, args):
        broadcasted_array = mx.zeros(broadcast_shape, dtype=arg.dtype)
        for i, dim in enumerate(shape):
            slices = [slice(0, dim) if j == i else slice(None) for j in range(ndim)]
            broadcasted_array[tuple(slices)] = arg
        broadcasted_arrays.append(broadcasted_array)
    return tuple(broadcasted_arrays)


def meshgrid(*args):
    shapes = [arg.size for arg in args]
    ndim = len(args)

    # Prepare shape tuples for output arrays
    output_shape = tuple([ndim] + shapes)

    # Create arrays to hold coordinate values
    output = [arg.reshape(-1, 1) for arg in args]
    output = broadcast_arrays(*output)

    # Stack arrays to obtain final result
    final_result = mx.zeros(output_shape, dtype=output[0].dtype)
    for i, arr in enumerate(output):
        final_result[i] = arr

    return tuple(final_result)


def dropout_fn(x: mx.array, p: float, training: bool) -> mx.array:
    """
    Dropout the input tensor.
    Args:
        x (Tensor): input tensor.
        p (float): Dropout probability.
        training (bool): Training flag.
    Returns:
        Tensor: The dropout tensor.
    """
    if p > 0 and training:
        mask = mx.random.uniform(0, 1, x.shape, dtype=x.dtype, ctx=x.context) > p
        x = x * mask / (1 - p)
    return x


def shifted_window_attention(
    input: mx.array,
    qkv_weight: mx.array,
    proj_weight: mx.array,
    relative_position_bias: mx.array,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[mx.array] = None,
    proj_bias: Optional[mx.array] = None,
    logit_scale: Optional[mx.array] = None,
    training: bool = True,
) -> mx.array:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, C, H, W]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, C, H, W]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = mx.pad(input, pad_width=[(0, 0), (0, pad_r), (0, pad_b), (0, 0)])
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = roll(x, shifts=(-shift_size[0], -shift_size[1]), axes=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.reshape(
        B,
        pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1],
        C,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(
        B * num_windows, window_size[0] * window_size[1], C
    )  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias
        length = qkv_bias.size // 3
        qkv_bias[length : 2 * length] = 0
    qkv = mx.matmul(x, qkv_weight.T) + qkv_bias
    qkv = qkv.reshape(x.shape[0], x.shape[1], 3, num_heads, C // num_heads).transpose(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = normalize(q, axis=-1) @ normalize(k, axis=-1).transpose(0, 1, 3, 2)
        logit_scale = mx.clip(logit_scale, a_min=None, a_max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = mx.zeros(
            (pad_H, pad_W), dtype=x.dtype
        )  # x.new_zeros((pad_H, pad_W))
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.reshape(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
        )
        attn_mask = attn_mask.transpose(0, 2, 1, 3).reshape(
            num_windows, window_size[0] * window_size[1]
        )
        attn_mask = attn_mask[:, None] - attn_mask[:, :, None]
        # attn_mask[attn_mask != 0] = -100.0
        attn_mask = mx.where(attn_mask != 0, -100.0, 0.0)
        attn = attn.reshape(
            x.shape[0] // num_windows, num_windows, num_heads, x.shape[1], x.shape[1]
        )
        attn = attn + attn_mask[None, :, None, ...]
        attn = attn.reshape(-1, num_heads, x.shape[1], x.shape[1])

    attn = nn.softmax(attn, axis=-1)
    attn = dropout_fn(attn, p=attention_dropout, training=training)

    x = mx.matmul(attn, v).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], C)
    x = mx.matmul(x, proj_weight.T) + proj_bias
    x = dropout_fn(x, p=dropout, training=training)

    # reverse windows
    x = x.reshape(
        B,
        pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = roll(x, shifts=(shift_size[0], shift_size[1]), axes=(1, 2))

    # unpad features
    x = x[:, :H, :W, :]
    return x


class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = mx.zeros(
            (
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.normal(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = mx.arange(self.window_size[0])
        coords_w = mx.arange(self.window_size[1])
        coords = mx.stack(meshgrid(coords_h, coords_w), axis=0)  # 2, Wh, Ww
        coords_flatten = mx.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.relative_position_index = relative_position_index

    def get_relative_position_bias(self) -> mx.array:
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            self.window_size,  # type: ignore[arg-type]
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x (Tensor): Tensor with layout of [B, C, H, W]
        Returns:
            Tensor with same layout as input, i.e. [B, C, H, W]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """
    See :func:`shifted_window_attention_v2`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = mx.log(10 * mx.ones((num_heads, 1, 1)))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, num_heads, bias=False),
        )
        if qkv_bias:
            length = self.qkv.bias.size // 3
            self.qkv.bias[length : 2 * length] = 0

    def define_relative_position_bias_table(self):
        # get relative_coords_table
        relative_coords_h = mx.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=mx.float32
        )
        relative_coords_w = mx.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=mx.float32
        )
        relative_coords_table = mx.stack(meshgrid(relative_coords_h, relative_coords_w))
        relative_coords_table = relative_coords_table.transpose(1, 2, 0)[
            None
        ]  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            mx.sign(relative_coords_table)
            * mx.log2(mx.abs(relative_coords_table) + 1.0)
            / 3.0
        )
        self.relative_coords_table = relative_coords_table

    def get_relative_position_bias(self) -> mx.array:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).reshape(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * mx.sigmoid(relative_position_bias)
        return relative_position_bias

    def __call__(self, x: mx.array):
        """
        Args:
            x (Tensor): Tensor with layout of [B, C, H, W]
        Returns:
            Tensor with same layout as input, i.e. [B, C, H, W]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=dropout
        )

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight)
                if m.bias is not None:
                    nn.init.normal(m.bias, std=1e-6)

    def __call__(self, x: mx.array):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlockV2(SwinTransformerBlock):
    """
    Swin Transformer V2 Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def __call__(self, x: mx.array):
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3,
                    embed_dim,
                    kernel_size=(patch_size[0], patch_size[1]),
                    stride=(patch_size[0], patch_size[1]),
                ),
                nn.Identity(),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.avgpool = AdaptiveAveragePool2D((1, 1))
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant(0)(m.bias)

    def __call__(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = mx.flatten(x, start_axis=1)
        x = self.head(x)
        return x

    def load_pytorch_weights(self, weights: Any) -> None:
        load_pytorch_weights(self, weights, ["features.0.0.weight"])


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    **kwargs: Any,
) -> SwinTransformer:
    model = SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )
    return model


@register_model()
def swin_t(pretrained: bool = True) -> SwinTransformer:
    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/swin_t-704ceda3.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def swin_s(pretrained: bool = True) -> SwinTransformer:
    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/swin_s-5e29d889.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def swin_b(pretrained: bool = True) -> SwinTransformer:
    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/swin_b-68c6b09e.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def swin_v2_t(pretrained: bool = True, **kwargs) -> SwinTransformer:
    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def swin_v2_s(pretrained: bool = True, **kwargs) -> SwinTransformer:
    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def swin_v2_b(pretrained: bool = True, **kwargs) -> SwinTransformer:
    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/swin_v2_b-781e5279.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model
