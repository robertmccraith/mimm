"""VGG
Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vgg.py
"""
from pathlib import Path
from typing import Union, List
import mlx.core as mx
import mlx.nn as nn
from mimm.layers.adaptive_average_pooling import AdaptiveAveragePool2D
from mimm.models.utils import get_pytorch_weights, load_pytorch_weights
from mimm.models._registry import register_model


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(
        self,
        cfg: List,
        num_classes: int = 1000,
        dropout: float = 0.5,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.features = make_layers(cfg, batch_norm=batch_norm)
        self.avgpool = AdaptiveAveragePool2D((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def load_pytorch_weights(self, weights_path: Path):
        load_pytorch_weights(
            self, weights_path, conv_layers=["features"], padding_layers=["features.20"]
        )
        classifier_in = self.classifier.layers[0].weight.shape[0]
        latent_shape = [512, 7, 7]
        self.classifier.layers[0].weight = (
            self.classifier.layers[0]
            .weight.reshape(classifier_in, *latent_shape)
            .transpose(0, 2, 3, 1)
            .reshape(classifier_in, -1)
        )


@register_model()
def vgg11(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], **kwargs
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg11-bbd30ac9.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vgg11_bn(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        batch_norm=True,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg11_bn-6002323d.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vgg13(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg13-c768596a.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vgg13_bn(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        batch_norm=True,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vgg16(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [
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
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vgg16_bn(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [
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
        batch_norm=True,
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vgg19(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [
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
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def vgg19_bn(pretrained: bool = True, **kwargs) -> VGG:
    model = VGG(
        [
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
        **kwargs,
    )
    if pretrained:
        weights_url = "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model
