from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union
import mlx.core as mx
import mlx.nn as nn
from mimm.layers.adaptive_average_pooling import AdaptiveAveragePool2D
from mimm.models.utils import get_pytorch_weights, load_pytorch_weights
from mimm.models._registry import register_model


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        # groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = AdaptiveAveragePool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)

        return x

    def features(self, x: mx.array) -> Dict[str, mx.array]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature_layers = {}
        x = self.layer1(x)
        feature_layers["layer1"] = x
        x = self.layer2(x)
        feature_layers["layer2"] = x
        x = self.layer3(x)
        feature_layers["layer3"] = x
        x = self.layer4(x)
        feature_layers["layer4"] = x

        return feature_layers

    def load_pytorch_weights(self, weights_path: Path):
        load_pytorch_weights(self, weights_path, ["conv", "downsample"])


TORCH_CACHE = Path().home() / ".cache/torch/hub/checkpoints"


@register_model()
def resnet18(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        weights_url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def resnet34(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights_url = "https://download.pytorch.org/models/resnet34-b627a593.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def resnet50(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        weights_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def resnet101(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        weights_url = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def resnet152(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        weights_url = "https://download.pytorch.org/models/resnet152-394f9c45.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def wide_resnet50_2(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-50-2 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2, **kwargs)
    if pretrained:
        weights_url = "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model


@register_model()
def wide_resnet101_2(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-101-2 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], width_per_group=64 * 2, **kwargs)
    if pretrained:
        weights_url = (
            "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth"
        )
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model
