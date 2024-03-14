# MLX Image Models
## Introduction
This repository contains implementations for image models in [MLX](https://github.com/ml-explore/mlx) with functions to load pretrained weights from PyTorch.
Example training code is provided in `mimm/scripts`.

## Installation
```bash
pip install mimm
```

## Usage
```python
from mimm import get_model, list_models

# List available models
print(list_models())

# Load a model and convert weights from torchvision
model = get_model('resnet50', pretrained=True)
```


## Models
### Classification
- [x] [AlexNet](https://arxiv.org/abs/1404.5997)
- [x] [VGG](https://arxiv.org/abs/1409.1556) + BatchNorm variants
- [x] [Resnet](https://arxiv.org/abs/1512.03385)
- [x] [Wide Resnet](https://arxiv.org/abs/1605.07146)
- [x] [MobileNetV3](https://arxiv.org/abs/1905.02244) 
- [x] [ViT](https://arxiv.org/abs/2010.11929)
- [x] [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [x] [Swin v2](https://arxiv.org/abs/2111.09883)

## Acknowledgements
This repository is based on the following repositories:
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models/tree/main)
- [Torchvision](https://github.com/pytorch/vision)
