import cv2
import numpy as np
from common.config import NOMINAL_SIZE
from train.preprocessing.raw import _downscale


def from_window(image: np.ndarray)->np.ndarray:
    image = cv2.resize(image, (NOMINAL_SIZE, NOMINAL_SIZE))
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def from_array(image: np.ndarray)->np.ndarray:
    return _downscale(image)
