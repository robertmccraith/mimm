from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
from mimm.models.resnet import BasicBlock, ResNet
from mimm.scripts.utils import eval_fn, plot_graphs
from mlx.data.datasets import load_imagenet
from mimm.scripts.validate import validate


def loss_fn(model, X, y):
    p = model(X)
    xe = nn.losses.cross_entropy(p, y)
    mx.simplify(xe)
    return mx.mean(xe)


def train(model, loss_and_grad_fn, train_dataloader, optimizer):
    model.train()
    losses = []
    accuracy = []
    progress = tqdm(
        enumerate(train_dataloader),
        desc="Training",
        ncols=80,
    )
    batch_size = -1
    for batch_idx, batch in progress:
        image = mx.array(batch["image"], dtype=mx.float32)
        label = mx.array(batch["label"])
        if batch_size == -1:
            batch_size = image.shape[0]
        elif batch_size != image.shape[0] or batch_idx < 3000:
            continue
        loss, grads = loss_and_grad_fn(model, image, label)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
        prog_bar = {"loss": np.mean(losses[-10:])}
        if batch_idx % 10 == 0:
            X = model(image)
            acc = eval_fn(X, label)
            accuracy.append(acc.item())
        if batch_idx % 500 == 0:
            print("loss", np.mean(losses[-10:]), "acc", np.mean(accuracy[-10:]))
        prog_bar["acc"] = np.mean(accuracy[-10:])
        progress.set_postfix(prog_bar)
    train_dataloader.reset()
    return losses, accuracy


def get_dataset(batch_size, root=None):
    mean = mx.array([0.485, 0.456, 0.406])
    std = mx.array([0.229, 0.224, 0.225])

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    tr = load_imagenet(root=root)
    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_resize_smallest_side("image", 256)
        .image_random_crop("image", 256, 256)
        .image_random_h_flip("image", prob=0.5)
        .image_random_crop("image", 224, 224)
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(prefetch_size=8, num_threads=8)
    )

    test = load_imagenet(root=root, split="val")
    test_iter = (
        test.to_stream()
        .image_resize("image", 256, 256)
        .image_center_crop("image", 224, 224)
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(prefetch_size=8, num_threads=8)
    )

    return tr_iter, test_iter


def main(data_path, epochs=100, eval_every=10, batch_size=256):
    train_iter, val_iter = get_dataset(batch_size, data_path)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    optimizer = optim.SGD(learning_rate=1e-1, momentum=0.9)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    train_losses = []
    train_accs = []
    for e in range(epochs):
        train_loss, train_acc = train(model, loss_and_grad_fn, train_iter, optimizer)
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        plot_graphs(train_losses, e + 1, "train", "loss")
        plot_graphs(train_accs, e + 1, "train", "accuracy")

        print(
            f"Epoch {e} train loss: {np.mean(train_loss)} train acc: {np.mean(train_acc)}"
        )
        model.save_weights(f"resnet_{e}")
        if e % eval_every == 0:
            val_acc = validate(model, val_iter)
            print(f"Epoch {e} val acc: {np.mean(val_acc)}")
            plot_graphs(val_acc, e + 1, "val", "accuracy")


if __name__ == "__main__":
    main(Path.home() / "imagenet")
