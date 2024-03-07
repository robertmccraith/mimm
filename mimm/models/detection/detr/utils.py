from typing import List, Optional
import mlx.nn as nn
import mlx.core as mx


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(
        self,
        num_features: int,
    ):
        super().__init__()

        self.num_features = num_features
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def __call__(self, x):
        print(x.min(), x.max())
        w = self.weight
        b = self.bias
        rv = self.running_var
        rm = self.running_mean
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[mx.array]):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[mx.array]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, h, w, c = batch_shape
        tensor = mx.zeros(batch_shape)
        mask = mx.zeros((b, h, w))
        for i in range(b):
            img = tensor_list[i]
            tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]] = img
            mask[i, : img.shape[1], : img.shape[2]] = True
        # for img, pad_img, m in zip(tensor_list, tensor, mask):
        # pad_img[:img.shape[0], : img.shape[1], : img.shape[2]] = img
        # m[:img.shape[1], :img.shape[2]] = True
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)
