from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mimm.layers.adaptive_average_pooling import AdaptiveAveragePool2D
from mimm.models.utils import get_pytorch_weights, load_pytorch_weights
from mimm.models._registry import register_model


class AlexNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        pytorch_weight_compatable=False,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = AdaptiveAveragePool2D((6, 6))
        self.pytorch_weight_compatable = pytorch_weight_compatable
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
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
            self, weights_path, conv_layers=["features"], padding_layers=["features.12"]
        )
        classifier_in = self.classifier.layers[1].weight.shape[0]
        latent_shape = [256, 6, 6]
        self.classifier.layers[1].weight = (
            self.classifier.layers[1]
            .weight.reshape(classifier_in, *latent_shape)
            .transpose(0, 2, 3, 1)
            .reshape(classifier_in, -1)
        )


@register_model()
def alexnet(pretrained: bool = True, **kwargs) -> AlexNet:
    model = AlexNet(**kwargs)
    if pretrained:
        weights_url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        weights = get_pytorch_weights(weights_url)
        model.load_pytorch_weights(weights)
    return model
