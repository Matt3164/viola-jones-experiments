import cv2
import numpy as np
from skimage.transform import integral_image

from common.config import IM_SIZE


def preprocess(impath: str)-> np.ndarray:
    image = cv2.imread(impath)
    image = cv2.resize(image, (IM_SIZE, IM_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return integral_image(image)