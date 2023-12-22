from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: int = 0,
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = [(0, 0), (padding, padding), (padding, padding), (0, 0)]

    def __call__(self, x: mx.array) -> mx.array:
        # padding
        B, H, W, C = x.shape
        x = mx.pad(x, pad_width=self.padding)

        ks = self.kernel_size
        stride = self.stride
        output_H = (H + sum(self.padding[1]) - ks[0]) // stride[0] + 1
        output_W = (W + sum(self.padding[2]) - ks[1]) // stride[1] + 1

        stride_steps_x = np.arange(0, H, stride[0]).repeat(output_W)
        stride_steps_y = np.tile(np.arange(0, H, stride[1]), output_H)
        stride_steps_x = mx.array(stride_steps_x).reshape(-1)
        stride_steps_y = mx.array(stride_steps_y).reshape(-1)
        x = [
            x[:, stride_steps_x + i, stride_steps_y + j].reshape(B, 1, -1, C)
            for i in range(ks[0])
            for j in range(ks[1])
        ]
        x = mx.concatenate(x, axis=1)
        x = x.reshape(B, ks[0] * ks[1], -1, C)
        x = x.max(axis=1).reshape(B, output_H, output_W, C)
        return x
