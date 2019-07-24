from itertools import chain
from typing import Iterator, Tuple, List
from numpy.core._multiarray_umath import ndarray

from pypurr.common.helpers.image import crop
from pypurr.train.helpers.dataset.objdet import Size2D, Rect
from pypurr.common.scanner import grid

# TODO use skimage sliding window implementation
def from_array(
        arr: ndarray,
        steps: List[Size2D],
        sizes: List[Size2D],
) -> Iterator[Tuple[Rect, ndarray]]:

    return chain.from_iterable(_from_array(arr, steps, sizes))

def _single_scale_from_array(img_arr: ndarray,
                             step: Size2D,
                             size: Size2D,
                             ) -> Iterator[Tuple[Rect, ndarray]]:
    return map(
        lambda bb: (bb, crop(img_arr, bb[0], bb[1])),
        grid.from_image(img_arr, size, step))


def _from_array(arr: ndarray, steps: List[Size2D], sizes: List[Size2D])->Iterator[Iterator[Tuple[Rect, ndarray]]]:
    for step, size in zip(steps, sizes):
        yield _single_scale_from_array(arr, step, size)



