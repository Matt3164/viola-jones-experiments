import cv2
from numpy.core._multiarray_umath import ndarray


def read_ima(fn: str)->ndarray:
    return cv2.cvtColor(
        cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)