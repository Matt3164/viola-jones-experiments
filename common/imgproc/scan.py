from itertools import chain
from typing import Iterator, Tuple, List

from numpy.core._multiarray_umath import ndarray, arange

# TODO use skimage sliding window implementation
from common.imgproc import grid
from common.iotools.image import crop


def from_image(
        img_arr: ndarray,
        step: int,
        size: int,
) -> Iterator[Tuple[Tuple[int, int, int, int], ndarray]]:
    return map(
        lambda bb: ((bb[0][0],bb[0][1],bb[1][0],bb[1][1]), crop(img_arr, bb[0], bb[1]) ),
        grid.from_image(img_arr, (size, size), (step, step)))

def from_image_pyramid(img: ndarray,
                       sizes: List[int],
                       steps: List[int]
                       )->Iterator[Tuple[Tuple[int, int, int, int], ndarray]]:
    assert len(sizes)==len(steps), "Not the same number of scales and steps"

    return chain.from_iterable(map(lambda ss: from_image(img, ss[1], ss[0]), zip(sizes, steps)))