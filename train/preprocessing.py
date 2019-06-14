import cv2
import numpy as np
from skimage.transform import integral_image

from common.config import IM_SIZE

def from_path(impath: str)-> np.ndarray:
    image = cv2.imread(impath)
    return from_array(image)


def from_array(image: np.ndarray)->np.ndarray:
    image = cv2.resize(image, (IM_SIZE, IM_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return integral_image(image)

