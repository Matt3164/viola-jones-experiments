from typing import Tuple

import cv2
import numpy as np
from numpy.core._multiarray_umath import ndarray


def from_path(fn: str)->ndarray:
    return cv2.cvtColor(
        cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)


def to_path(fn: str, arr: ndarray)->bool:
    return cv2.imwrite(
        fn,
        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    )


def _crop(arr: np.ndarray, bb: Tuple[int, int, int, int])->np.ndarray:
    return crop(arr, (bb[0], bb[1]), (bb[2], bb[3]))

def crop(arr: np.ndarray, top_left: Tuple[int,int], size: Tuple[int,int])->np.ndarray:
    return arr[top_left[0]:(top_left[0] + size[0]), top_left[1]:(top_left[1] + size[1]), :]
