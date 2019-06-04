from typing import Tuple
import numpy as np


def _bbox_to_mask(bbox: Tuple[int, int, int, int], shape: Tuple[int, int], burn_values=1):
    coords = np.clip(bbox, a_min=0, a_max=max(shape))

    array = np.zeros(shape, dtype=np.uint8)

    array[coords[2]:coords[3], coords[0]:coords[1]] = burn_values

    return array
