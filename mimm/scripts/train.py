from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from mimm.datasets.cifar100 import CIFAR100
from mimm.models.vgg import VGG
from mimm.transforms.transforms import (
    Composer,
    Normalize,
    RandomHorizontalFlip,
)
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d


def collate_fn(batch):
    images, labels = list(zip(*batch))
    images = mx.array(np.stack(images), dtype=mx.float32)
    labels = mx.array(labels, dtype=mx.int32)
    return images, labels


def loss_fn(model, X, y):
    p = model(X)
    xe = nn.losses.cross_entropy(p, y)
    mx.simplify(xe)
    return mx.mean(xe)


def eval_fn(X, y):
    return mx.mean(mx.argmax(X, axis=1) == y)


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


def train(model, loss_and_grad_fn, train_dataloader, optimizer):
    model.train()
    losses = []
    accuracy = []
    progress = tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        ncols=80,
    )
    for batch_idx, batch in progress:
        image, label = collate_fn(batch)
        loss, grads = loss_and_grad_fn(model, image, label)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        # print(model.features.layers[0].weight[0, 0, 0, 0], grads["features"]["layers"][0]["weight"][0, 0, 0, 0])
        losses.append(loss.item())
        prog_bar = {"loss": np.mean(losses[-10:])}
        if batch_idx % 10 == 0:
            X = model(image)
            acc = eval_fn(X, label)
            accuracy.append(acc.item())
        if batch_idx % 100 == 0:
            print("loss", np.mean(losses[-10:]), "acc", np.mean(accuracy[-10:]))
        prog_bar["acc"] = np.mean(accuracy[-10:])
        progress.set_postfix(prog_bar)
    return losses, accuracy


def validate(model, loss_and_grad_fn, val_dataloader):
    model.eval()
    losses = []
    accuracy = []
    progress = tqdm(
        enumerate(val_dataloader),
        desc="Validation",
        total=len(val_dataloader),
        ncols=80,
    )
    for batch_idx, batch in progress:
        image, label = collate_fn(batch)
        X = model(image)
        loss, grads = loss_and_grad_fn(X, label)
        losses.append(loss.item())
        acc = eval_fn(X, label)
        accuracy.append(acc.item())
        progress.set_postfix({"loss": np.mean(losses), "acc": np.mean(accuracy)})
    return losses, accuracy


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


def main(data_path, epochs=100, eval_every=10, batch_size=256):
    dataset_class = CIFAR100
    train_transforms, val_transforms = cifar_transforms()
    train_dataset = dataset_class(data_path, split="train", transform=train_transforms)
    val_dataset = dataset_class(data_path, split="test", transform=train_transforms)

    def identity(x):
        return x

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=identity,
        shuffle=True,
        num_workers=16,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=identity, num_workers=8
    )
    model = VGG(VGG.cfgs["vgg11"], num_classes=100, batch_norm=True)
    optimizer = optim.SGD(learning_rate=1e-1, momentum=0.9)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    train_losses = []
    train_accs = []
    for e in range(epochs):
        train_loss, train_acc = train(
            model, loss_and_grad_fn, train_dataloader, optimizer
        )
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        plot_graphs(train_losses, train_accs, e + 1, "train")

        print(
            f"Epoch {e} train loss: {np.mean(train_loss)} train acc: {np.mean(train_acc)}"
        )
        model.save_weights(f"vgg11_{e}")
        if e % eval_every == 0:
            val_loss, val_acc = validate(model, loss_and_grad_fn, val_dataloader)
            print(
                f"Epoch {e} val loss: {np.mean(val_loss)} val acc: {np.mean(val_acc)}"
            )
            plot_graphs(val_loss, val_acc, e + 1, "val")


if __name__ == "__main__":
    main(Path("cifar100"))
