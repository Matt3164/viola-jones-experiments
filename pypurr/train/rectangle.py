from typing import Tuple
import numpy as np


def rect_to_mask(rect: Tuple[int, int, int, int], shape: Tuple[int, int], burn_values: int=1)-> np.ndarray:

    """

    Convert a rectangle to a mask i.e. binary image.

    Args:
        rect: (imin,imax,jmin,jmax)
        shape:
        burn_values: default value inside the rectangle in the mask.

    Returns:

    """


    coords = np.clip(rect, a_min=0, a_max=max(shape))

    array = np.zeros(shape, dtype=np.uint8)

    array[coords[2]:coords[3], coords[0]:coords[1]] = burn_values

    return array