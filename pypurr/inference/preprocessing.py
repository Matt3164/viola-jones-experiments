import cv2
import numpy as np
from skimage.transform import integral_image

from pypurr.common.config import NOMINAL_SIZE
from pypurr.train.preprocessing.raw import _downscale


def from_window(image: np.ndarray)->np.ndarray:
    image = cv2.resize(image, (NOMINAL_SIZE, NOMINAL_SIZE))
    return integral_image(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def from_array(image: np.ndarray)->np.ndarray:
    return _downscale(image)
