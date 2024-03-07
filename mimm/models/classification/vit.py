from functools import partial
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn


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
        self.self_attention = nn.MultiHeadAttention(hidden_dim, num_heads)
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
