from pathlib import Path
import pickle


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class CIFAR100:
    def __init__(self, root: Path, split="train", transform=None):
        super().__init__()
        self.transform = transform
        data = unpickle(root / split)
        self.images = data[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = data[b"fine_labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ind):
        img = self.images[ind]
        cls = self.labels[ind]
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
