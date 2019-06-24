import cv2
from numpy.core._multiarray_umath import ndarray


def from_path(fn: str)->ndarray:
    return cv2.cvtColor(
        cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)


def to_path(fn: str, arr: ndarray)->bool:
    return cv2.imwrite(
        fn,
        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    )