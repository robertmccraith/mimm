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
            Resize(256),
            RandomCrop(224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transforms, val_transforms


def plot_graphs(losses, accuracies, epoch, name):
    plt.clf()
    xs = np.arange(len(losses)) / (len(losses) / epoch)
    plt.plot(xs, losses)
    smooth = gaussian_filter1d(losses, sigma=0.1 * len(losses) / epoch)
    plt.plot(xs, smooth)
    plt.title("Loss")
    plt.savefig(f"{name}-loss.png")
    plt.clf()
    xs = np.arange(len(accuracies)) / (len(accuracies) / epoch)
    plt.plot(xs, accuracies)
    smooth = gaussian_filter1d(accuracies, sigma=0.1 * len(accuracies) / epoch)
    plt.plot(xs, smooth)
    plt.title("Accuracy")
    plt.savefig(f"{name}-accuracy.png")


def collate_fn(batch):
    images, labels = list(zip(*batch))
    images = mx.array(np.stack(images), dtype=mx.float32)
    labels = mx.array(labels, dtype=mx.int32)
    return images, labels


def eval_fn(X, y):
    return mx.mean(mx.argmax(X, axis=1) == y)


def identity(x):
    return x
