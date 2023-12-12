from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = (
            (padding, padding) if isinstance(padding, int) else (padding[0], padding[1])
        )

    def __call__(self, x: mx.array) -> mx.array:
        # padding
        x = mx.pad(x, pad_width=self.padding)
        B, H, W, C = x.shape

        # max pool by reshaping
        ks = self.kernel_size
        x = x.reshape(
            B,
            H // ks[0],
            ks[0],
            W // ks[1],
            ks[1],
            C,
        )
        x = mx.max(x, axis=(2, 4))
        return x


if __name__ == "__main__":
    # t = mx.arange(32 * 224 * 224 * 64).reshape(32, 224, 224, 64)
    t = mx.arange(16).reshape(1, 4, 4, 1)
    bn = MaxPool2d(2, 2, 0)
    print(t[..., 0])
    print(bn(t))
