from glob import glob
from os.path import join
from typing import List

import cv2
from numpy.core._multiarray_umath import ndarray


def read(fn: str)->ndarray:
    return cv2.cvtColor(
        cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

def write(fn: str, arr: ndarray)->bool:
    return cv2.imwrite(
        fn,
        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    )


def glob_png(path: str)->List[str]:
    return glob(join(path, "*.png"))