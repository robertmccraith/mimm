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
        x = mx.concatenate(
            [
                mx.concatenate(
                    [
                        x[:, j : j + ks[0], i : i + ks[1], :].max(
                            axis=(1, 2), keepdims=True
                        )
                        for i in range(0, W, self.stride[1])
                        if i + ks[1] <= W
                    ],
                    axis=2,
                )
                for j in range(0, H, self.stride[0])
                if j + ks[0] <= H
            ],
            axis=1,
        )

        return x


if __name__ == "__main__":
    s = 13
    t = mx.arange(s**2).reshape(1, s, s, 1)
    mp = MaxPool2d(3, 2, 0)
    print(t[..., 0])
    pooled = mp(t)
    print(pooled[..., 0])
    print(pooled.shape)
