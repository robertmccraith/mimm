import math
from typing import Optional
import mlx.core as mx
import mlx.nn as nn


def _canonical_mask(
    mask: Optional[mx.array],
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    return (mask).astype(dtype) * -1e9


class MultiHeadAttention(nn.Module):
    """Implements the scaled dot product attention with multiple heads.

    Given inputs for queries, keys and values the ``MultiHeadAttention``
    produces new values by aggregating information from the input values
    according to the similarities of the input queries and keys.

    All inputs as well as the output are linearly projected without biases by
    default.

    ``MultiHeadAttention`` also takes an optional additive attention mask that
    should be broadcastable with ``(batch, num_heads, # queries, # keys)``. The
    mask should have ``-inf`` or very large negative numbers at the positions
    that should *not* be attended to.

    Args:
        dims (int): The model dimensions. This is also the default
            value for the queries, keys, values, and the output.
        num_heads (int): The number of attention heads to use.
        query_input_dims (int, optional): The input dimensions of the queries.
            Default: ``dims``.
        key_input_dims (int, optional): The input dimensions of the keys.
            Default: ``dims``.
        value_input_dims (int, optional): The input dimensions of the values.
            Default: ``key_input_dims``.
        value_dims (int, optional): The dimensions of the values after the
            projection. Default: ``dims``.
        value_output_dims (int, optional): The dimensions the new values will
            be projected to. Default: ``dims``.
        bias (bool, optional): Whether or not to use a bias in the projections.
            Default: ``False``.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.query_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.key_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.value_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None, key_padding_mask=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys
        if key_padding_mask is not None:
            key_padding_mask = mx.repeat(
                key_padding_mask.reshape(B, 1, 1, S), num_heads, axis=1
            )
            key_padding_mask = _canonical_mask(key_padding_mask)
            mask = key_padding_mask if mask is None else mask + key_padding_mask
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).reshape(B, L, -1)

        return self.out_proj(values_hat)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype: mx.Dtype = mx.float32):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        # usually inf but 1e9 is as good and softmax(full(1e9)) != nan
        # TODO: Should replace this with finfo(dtype).min
        mask = mask.astype(dtype) * -1e9
        return mask
