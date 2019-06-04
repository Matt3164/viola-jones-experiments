from typing import Tuple

import cv2
import numpy as np


def _overlay_bbox_on_img(
        tup: Tuple[np.ndarray, Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (255, 0, 0)):
    array, bbox = tup
    coords = np.clip(bbox, a_min=0, a_max=max(array.shape))
    cv2.rectangle(array, (coords[0], coords[2]), (coords[1], coords[3]), color, 10)
    return array
