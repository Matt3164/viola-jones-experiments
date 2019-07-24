import cv2
import numpy as np
from numpy.core._multiarray_umath import ndarray
from glob import glob
from os.path import join
from typing import List

from pypurr.train.helpers.dataset.objdet import Point, Size2D


def from_path(fn: str)->ndarray:
    return cv2.cvtColor(
        cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)


def to_path(fn: str, arr: ndarray)->bool:
    return cv2.imwrite(
        fn,
        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    )

def crop(arr: np.ndarray, top_left: Point, size: Size2D)->np.ndarray:
    return arr[top_left[0]:(top_left[0] + size[0]), top_left[1]:(top_left[1] + size[1]), :]


def find_pngs(path: str)->List[str]:
    return glob(join(path, "*.png"))


def downscale(img: ndarray, target_size: int=256):
    while img.shape[0] > target_size and img.shape[1] > target_size:
        img = cv2.pyrDown(img)
    return img