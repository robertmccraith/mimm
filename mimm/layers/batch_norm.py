from typing import Tuple
import mlx.core as mx
import mlx.nn as nn


class BatchNorm(nn.Module):
    r"""Applies Batch Normalization over a 2D or 3D input.
    Computes
    .. math::
        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,
    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.
    [1]: https://arxiv.org/abs/1502.03167
    Args:
        num_features (int): The feature dimension of the input to normalize over.
        eps (float, optional): A small additive constant for numerical stability. Default is 1e-5.
        momentum (float, optional): The momentum for updating the running mean and variance. Default is 0.1.
        affine (bool, optional): If True, learn an affine transform to apply after the normalization. Default is True.
    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> mx.random.seed(42)
        >>> input = mx.random.normal((5, 4), dtype=mx.float32)
        >>> # Batch norm
        >>> bn = nn.BatchNorm(num_features=4, affine=True)
        >>> output = bn(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._dims_expanded = False

        if self.affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

        if self.track_running_stats:
            self._running_mean = mx.zeros((num_features,))
            self._running_var = mx.ones((num_features,))

    def _extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={'weight' in self}, track_running_stats={self.track_running_stats}"

    def _check_and_expand_dims(self, x: mx.array):
        """
        Check if the input is a 2D or 3D tensor and expand the weight, bias, running mean, and running variance accordingly.
        Args:
            x (mx.array): Input tensor.
        """

        num_dims = len(x.shape)
        dims_dict = {
            2: ((1, self.num_features), (0,)),
            3: ((1, 1, self.num_features), (0, 1)),
            4: ((1, 1, 1, self.num_features), (0, 1, 2)),
        }

        if num_dims not in dims_dict:
            raise ValueError(f"expected num_dims to be 2, 3, or 4 (got {num_dims})")

        shape, self.reduction_axes = dims_dict[num_dims]

        if self.affine:
            self.weight = mx.expand_dims(self.weight, self.reduction_axes)
            self.bias = mx.expand_dims(self.bias, self.reduction_axes)

        if self.track_running_stats:
            self._running_mean = mx.expand_dims(self._running_mean, self.reduction_axes)
            self._running_var = mx.expand_dims(self._running_var, self.reduction_axes)

        self._dims_expanded = True

    def _calc_stats(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Calculate the mean and variance of the input tensor.
        Args:
            x (mx.array): Input tensor.
        Returns:
            tuple: Tuple containing mean and variance.
        """

        means = mx.mean(x, axis=self.reduction_axes, keepdims=True)
        var = mx.var(x, axis=self.reduction_axes, keepdims=True)

        if self.track_running_stats and self.training:
            self._running_mean = (
                1 - self.momentum
            ) * self._running_mean + self.momentum * means
            self._running_var = (
                1 - self.momentum
            ) * self._running_var + self.momentum * var
        return means, var

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of BatchNorm.
        Args:
            x (mx.array): Input tensor.
        Returns:
            mx.array: Output tensor.
        """

        if not self._dims_expanded:
            self._check_and_expand_dims(x)

        if self.training or not self.track_running_stats:
            means, var = self._calc_stats(x)
        else:
            means, var = self._running_mean, self._running_var
        x = (x - means) * mx.rsqrt(var + self.eps)
        return (self.weight * x + self.bias) if "weight" in self else x
