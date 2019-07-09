import cv2
from numpy.core.multiarray import ndarray
from pypurr.common.config import PROCESS_SIZE

def _downscale(img: ndarray)->ndarray:

    while img.shape[0]>PROCESS_SIZE and img.shape[1]>PROCESS_SIZE:

        img = cv2.pyrDown(img)

    return img

def from_array(img: ndarray)->ndarray:
    return _downscale(img)


