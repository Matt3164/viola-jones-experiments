from itertools import chain
from typing import Tuple, List, Iterator
from numpy.core.multiarray import ndarray, arange


def scan_image(
        img_arr: ndarray,
        step: int,
        size: int,
) -> Iterator[Tuple[Tuple[int, int, int, int], ndarray]]:
    for i in arange(0, img_arr.shape[0] - size, step):
        for j in arange(0, img_arr.shape[1] - size, step):
            yield (i, j, size, size), img_arr[i:i + size, j:j + size, :]

def scan_pyramid_image(img: ndarray,
                       sizes: List[int],
                       steps: List[int]
                       )->Iterator[Tuple[Tuple[int, int, int, int], ndarray]]:
    assert len(sizes)==len(steps), "Not the same number of scales and steps"

    return chain.from_iterable(map( lambda ss: scan_image(img, ss[1], ss[0]), zip(sizes, steps) ))