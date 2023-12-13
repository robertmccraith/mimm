from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mimm.layers.adaptive_average_pooling import AdaptiveAveragePool2D
from mimm.layers.max_pool import MaxPool2d


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
            MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            MaxPool2d(kernel_size=3, stride=2),
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
        if self.pytorch_weight_compatable:
            x = x.transpose(0, 3, 1, 2)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def load_pytorch_weights(self, weights_path: Path):
        import torch

        self.pytorch_weight_compatable = True
        weights = torch.load(weights_path, map_location="cpu")
        for i, layer in enumerate(self.features.layers):
            if isinstance(layer, nn.Conv2d):
                layer.weight = mx.array(
                    weights[f"features.{i}.weight"].detach().cpu().numpy(),
                    # dtype=mx.float16,
                ).transpose(0, 2, 3, 1)
                layer.bias = mx.array(
                    weights[f"features.{i}.bias"].detach().cpu().numpy(),
                    # dtype=mx.float16,
                )
        for i, layer in enumerate(self.classifier.layers):
            if isinstance(layer, nn.Linear):
                layer.weight = mx.array(
                    weights[f"classifier.{i}.weight"].detach().cpu().numpy(),
                    # dtype=mx.float16,
                )
                layer.bias = mx.array(
                    weights[f"classifier.{i}.bias"].detach().cpu().numpy(),
                    # dtype=mx.float16,
                )
