from pathlib import Path
from PIL import Image
import numpy as np


class ImageNet:
    def __init__(self, root: Path, split="train", transform=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        class_folders = list((self.root / split).glob("*"))
        class_folders = sorted(class_folders)
        self.class_to_idx = {c.name: i for i, c in enumerate(class_folders)}
        self.image_files = [
            a
            for class_folder in class_folders
            for a in class_folder.glob("*")
            if a.suffix in [".JPEG"]
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, ind):
        img_file = self.image_files[ind]
        cls = self.class_to_idx[img_file.parent.name]
        img = Image.open(img_file).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
