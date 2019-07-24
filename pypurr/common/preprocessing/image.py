from numpy.core.multiarray import ndarray

from pypurr.common.config import DWNSCALE_SIZE
from pypurr.common.helpers.image import downscale


def from_array(arr: ndarray)->ndarray:
    return downscale(arr, target_size=DWNSCALE_SIZE)