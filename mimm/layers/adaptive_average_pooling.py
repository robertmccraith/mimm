from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn


def adaptive_average_pool1d(
    x: mx.array,
    output_size: int,
) -> mx.array:
    B, H, C = x.shape
    x = x.reshape(B, H // output_size, output_size, C)
    x = mx.mean(x, axis=2)
    return x.squeeze()


class AdaptiveAveragePool1D(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_average_pool1d(x, self.output_size)


def adaptive_average_pool2d(
    x: mx.array,
    output_size: tuple,
) -> mx.array:
    B, H, W, C = x.shape
    x = x.reshape(
        B, H // output_size[0], output_size[0], W // output_size[1], output_size[1], C
    )
    x = mx.mean(x, axis=(1, 3))
    return x


class AdaptiveAveragePool2D(nn.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.output_size = (
            output_size
            if isinstance(output_size, tuple)
            else (output_size, output_size)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return adaptive_average_pool2d(x, self.output_size)


if __name__ == "__main__":
    t = mx.arange(16).reshape(1, 4, 4, 1)
    print(t[..., 0])
    bn = AdaptiveAveragePool2D(2)
    print(bn(t)[..., 0])
