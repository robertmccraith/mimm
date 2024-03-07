from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import numpy as np
from mimm.transforms.transforms import (
    Composer,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
import mlx.core as mx


def cifar_transforms():
    train_transforms = Composer(
        [
            RandomHorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = Composer(
        [
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


def imagenet_transforms():
    train_transforms = Composer(
        [
            Resize(256),
            RandomCrop(224),
            RandomHorizontalFlip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = Composer(
        [
            Resize(800),
            # RandomCrop(800),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


def plot_graphs(ys, epoch, split, name):
    plt.clf()
    xs = np.arange(len(ys)) / (len(ys) / epoch)
    plt.plot(xs, ys)
    smooth = gaussian_filter1d(ys, sigma=0.1 * len(ys) / epoch)
    plt.plot(xs, smooth)
    plt.title(name)
    plt.savefig(f"{split}-{name}.png")


def eval_fn(X, y):
    return mx.mean(mx.argmax(X, axis=1) == y)
