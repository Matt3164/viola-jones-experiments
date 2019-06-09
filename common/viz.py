from typing import Tuple

import cv2
import numpy as np


def overlay_bbox_on_img(
        tup: Tuple[np.ndarray, Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (255, 0, 0))->np.ndarray:
    array, bbox = tup
    return overlay_bbox_on_array(array, bbox, color=color)

def overlay_bbox_on_array(
        img: np.ndarray,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (255, 0, 0))->np.ndarray:

    # In case of negative coordinates
    coords = np.clip(rect, a_min=0, a_max=max(img.shape))
    cv2.rectangle(img, (coords[0], coords[2]), (coords[1], coords[3]), color, 10)
    return img
