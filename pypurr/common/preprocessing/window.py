import cv2
import numpy as np
from skimage.transform import integral_image

from pypurr.common.config import NOMINAL_SIZE


def from_array(image: np.ndarray)->np.ndarray:
    image = cv2.resize(image, (NOMINAL_SIZE, NOMINAL_SIZE))
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # return integral_image(image)