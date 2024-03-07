from pathlib import Path
import numpy as np
from tqdm import tqdm
from mimm.models.resnet import BasicBlock, ResNet
from mimm.scripts.utils import eval_fn
from mlx.data.datasets import load_imagenet
import mlx.core as mx


def get_dataset(batch_size, root=None):
    mean = mx.array([0.485, 0.456, 0.406])
    std = mx.array([0.229, 0.224, 0.225])

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    test = load_imagenet(root=root, split="val")
    test_iter = (
        test.to_stream()
        .image_resize("image", 256, 256)
        .image_center_crop("image", 224, 224)
        .key_transform("image", normalize)
        .batch(batch_size)
    )

    return test_iter


def validate(model, val_dataloader):
    model.eval()
    accuracy = []
    progress = tqdm(
        enumerate(val_dataloader),
        desc="Validation",
        ncols=80,
    )
    batch_size = -1
    for batch_idx, batch in progress:
        image = mx.array(batch["image"], dtype=mx.float32)
        label = mx.array(batch["label"])
        if batch_size == -1:
            batch_size = image.shape[0]
        elif batch_size != image.shape[0]:
            continue
        X = model(image)
        acc = eval_fn(X, label)
        accuracy.append(acc.item())
        progress.set_postfix({"acc": np.mean(accuracy)})
    val_dataloader.reset()
    return accuracy


def main(data_path):
    val_iter = get_dataset(16, root=data_path)
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.load_pytorch_weights("resnet18.pth")
    model.eval()
    val_acc = validate(model, val_iter)
    print(f"Validation accuracy: {np.mean(val_acc)}")


if __name__ == "__main__":
    main(Path.home() / "imagenet")
