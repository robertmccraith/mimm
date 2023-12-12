import mlx.core as mx
import mlx.nn as nn
import numpy as np


class BatchNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.weight = mx.ones(dim)
        self.bias = mx.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdims=True)
            xvar = x.var(0, keepdims=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / mx.sqrt(xvar + self.eps)
        self.out = self.weight * xhat + self.bias
        # update the buffers
        if self.training:
            self.running_mean = np.array(
                (
                    (1 - self.momentum) * self.running_mean + self.momentum * xmean
                ).tolist()
            )
            self.running_var = np.array(
                ((1 - self.momentum) * self.running_var + self.momentum * xvar).tolist()
            )
        return self.out


if __name__ == "__main__":
    t = mx.random.uniform(shape=(2, 224, 224, 3))
    bn = BatchNorm2d(3)
    bn(t)
