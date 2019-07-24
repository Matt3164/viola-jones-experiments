from typing import Tuple

import cv2
import numpy as np

def draw_bbox_on_image(
        img: np.ndarray,
        rect: Tuple[Tuple[int, int], Tuple[int, int]],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness=2)->np.ndarray:

    cv2.rectangle(img, (rect[0][1], rect[0][0]), (rect[0][1]+rect[1][1], rect[0][0]+rect[1][0]), color, thickness)
    return img
