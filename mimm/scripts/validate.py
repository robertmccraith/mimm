from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from mimm.datasets.imagenet import ImageNet
from mimm.models.vgg import VGG
from mimm.scripts.utils import collate_fn, eval_fn, identity, imagenet_transforms


def validate(model, val_dataloader):
    model.eval()
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
        acc = eval_fn(X, label)
        accuracy.append(acc.item())
        progress.set_postfix({"acc": np.mean(accuracy)})
    return accuracy


def main(data_path):
    _, val_transforms = imagenet_transforms()
    val_dataset = ImageNet(data_path, split="val", transform=val_transforms)

    val_dataloader = DataLoader(
        val_dataset, batch_size=32, collate_fn=identity, num_workers=8, shuffle=True
    )
    model = VGG(VGG.cfgs["vgg11"])
    model.load_pytorch_weights("vgg11.pth")
    model.eval()
    val_acc = validate(model, val_dataloader)
    print(f"Validation accuracy: {np.mean(val_acc)}")


if __name__ == "__main__":
    main(Path("/Users/robertmccraith/imagenet"))
