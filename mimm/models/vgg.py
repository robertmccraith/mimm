"""VGG

Adapted from https://github.com/pytorch/vision 'vgg.py' (BSD-3-Clause) with a few changes for
timm functionality.

Copyright 2021 Ross Wightman
"""
from pathlib import Path
from typing import Union, List, Dict
import mlx.core as mx
import mlx.nn as nn
from mimm.layers.adaptive_average_pooling import AdaptiveAveragePool2D

from mimm.layers.batch_norm import BatchNorm2d
from mimm.layers.max_pool import MaxPool2d


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


CFGS: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    cfgs = CFGS

    def __init__(
        self,
        cfg: List,
        num_classes: int = 1000,
        dropout: float = 0.5,
        pytorch_weight_compatable: bool = False,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.features = make_layers(cfg, batch_norm=batch_norm)
        self.avgpool = AdaptiveAveragePool2D((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self.pytorch_weight_compatable = pytorch_weight_compatable

    def __call__(self, x: mx.array) -> mx.array:
        x = self.features(x)
        x = self.avgpool(x)
        if self.pytorch_weight_compatable:
            x = x.transpose(0, 3, 1, 2)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def load_pytorch_weights(self, weights_path: Path):
        import torch

        weights = torch.load(weights_path, map_location="cpu")
        for i, layer in enumerate(self.features.layers):
            if isinstance(layer, nn.Conv2d):
                layer.weight = mx.array(
                    weights[f"features.{i}.weight"].detach().cpu().numpy(),
                    dtype=mx.float16,
                ).transpose(0, 2, 3, 1)
                layer.bias = mx.array(
                    weights[f"features.{i}.bias"].detach().cpu().numpy(),
                    dtype=mx.float16,
                )
        for i, layer in enumerate(self.classifier.layers):
            if isinstance(layer, nn.Linear):
                layer.weight = mx.array(
                    weights[f"classifier.{i}.weight"].detach().cpu().numpy(),
                    dtype=mx.float16,
                )
                layer.bias = mx.array(
                    weights[f"classifier.{i}.bias"].detach().cpu().numpy(),
                    dtype=mx.float16,
                )

    @classmethod
    def from_pytorch_weights(
        cls,
        cfg: List,
        weights_path: Path,
        num_classes: int = 1000,
        dropout: float = 0.5,
    ) -> nn.Module:
        """Load a VGG model from a PyTorch weights file."""
        model = VGG(cfg, num_classes, dropout, pytorch_weight_compatable=True)
        model.load_pytorch_weights(weights_path)
        return model
