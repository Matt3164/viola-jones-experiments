from typing import Tuple

from cv2 import resize
from numpy.core.multiarray import ndarray, arange


def scan_image(
        img_arr: ndarray,
        step: int,
        size: int,
) -> Tuple[Tuple[int, int, int, int], ndarray]:
    for i in arange(0, img_arr.shape[0] - size, step):
        for j in arange(0, img_arr.shape[1] - size, step):
            yield (i, j, size, size), img_arr[i:i + size, j:j + size, :]
