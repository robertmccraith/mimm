from functools import partial
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mimm.models._registry import register_model
from mimm.models.utils import get_pytorch_weights, load_pytorch_weights


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        layers = []
        layers.append(nn.Linear(in_dim, mlp_dim, bias=True))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(mlp_dim, in_dim, bias=True))
        layers.append(nn.Dropout(dropout))

        super().__init__(*layers)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiHeadAttention(hidden_dim, num_heads, bias=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def __call__(self, input: mx.array) -> mx.array:
        assert (
            input.ndim == 3
        ), f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        x = self.ln_1(input)
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.pos_embedding = mx.zeros((1, seq_length, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        layers = []
        for i in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    norm_layer,
                )
            )
        self.layers = nn.Sequential(*layers)
        self.ln = norm_layer(hidden_dim)

    def __call__(self, input: mx.array) -> mx.array:
        assert (
            input.ndim == 3
        ), f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
        input = input + self.pos_embedding
        x_ = self.layers(self.dropout(input))
        return self.ln(x_)


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Input shape indivisible by patch size!"
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = mx.zeros((1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers = []
        if representation_size is None:
            heads_layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            heads_layers.append(nn.Linear(hidden_dim, representation_size))
            heads_layers.append(nn.Tanh())
            heads_layers.append(nn.Linear(representation_size, num_classes))

        self.heads = nn.Sequential(*heads_layers)

    def _process_input(self, x: mx.array) -> mx.array:
        n, h, w, c = x.shape
        p = self.patch_size
        assert (
            h == self.image_size
        ), f"Wrong image height! Expected {self.image_size} but got {h}!"
        assert (
            w == self.image_size
        ), f"Wrong image width! Expected {self.image_size} but got {w}!"
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(n, n_h * n_w, self.hidden_dim)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = mx.repeat(self.class_token, n, 0)
        x = mx.concatenate([batch_class_token, x], axis=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

    def load_pytorch_weights(self, weights_path: str):
        def layer_name_changes(k, v):
            if "encoder_layer_" in k:
                k = k.replace("encoder_layer_", "encoder.layers.")
            if "layers.encoder.layers" in k:
                k = k.replace("layers.encoder.layers", "layers")
            if "in_proj_weight" in k:
                output = []
                query, key, value = mx.split(v, 3, axis=0)
                output.append((k.replace("in_proj_weight", "query_proj.weight"), query))
                output.append((k.replace("in_proj_weight", "key_proj.weight"), key))
                output.append((k.replace("in_proj_weight", "value_proj.weight"), value))
                return output
            if "in_proj_bias" in k:
                output = []
                query, key, value = mx.split(v, 3, axis=0)
                output.append((k.replace("in_proj_bias", "query_proj.bias"), query))
                output.append((k.replace("in_proj_bias", "key_proj.bias"), key))
                output.append((k.replace("in_proj_bias", "value_proj.bias"), value))
                return output
            if "linear_1" in k:
                k = k.replace("linear_1", "0")
            if "linear_2" in k:
                k = k.replace("linear_2", "3")
            if "heads.head" in k:
                k = k.replace("heads.head", "heads.layers")
            return [(k, v)]

        def layer_modify(weights):
            weights["heads"] = {"layers": [weights["heads"]["layers"]]}

        load_pytorch_weights(
            self,
            weights_path,
            conv_layers=["conv_proj"],
            layer_name_changes=layer_name_changes,
            layer_modify=layer_modify,
        )


@register_model()
def vit_b_16(pretrained: bool = True, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vit_b_16-c867db91.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vit_b_32(pretrained: bool = True, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vit_l_16(pretrained: bool = True, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vit_l_32(pretrained: bool = True, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vit_l_32-c7638314.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vit_h_14(pretrained: bool = True, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        image_size=224,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        **kwargs,
    )
    if pretrained:
        raise NotImplementedError("Pretrained weights not available for this model!")
    return model
