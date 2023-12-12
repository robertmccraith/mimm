from numpy.typing import NDArray
from typing import List, Tuple, Union
import cv2
import numpy as np


class Resize:
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image: NDArray) -> NDArray:
        image = cv2.resize(image, self.size)
        return image


class RandomCrop:
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image: NDArray) -> NDArray:
        h, w = image.shape[:2]
        new_h, new_w = self.size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top : top + new_h, left : left + new_w]
        return image


class CenterCrop:
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image: NDArray) -> NDArray:
        h, w = image.shape[:2]
        new_h, new_w = self.size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        image = image[top : top + new_h, left : left + new_w]
        return image


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: NDArray) -> NDArray:
        if np.random.random() < self.p:
            image = cv2.flip(image, 1)
        return image


class Normalize:
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ):
        self.mean = mean
        self.std = std

    def __call__(self, image: NDArray) -> NDArray:
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image


class Composer:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: NDArray) -> NDArray:
        for t in self.transforms:
            image = t(image)
        return image
