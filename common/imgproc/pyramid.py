from itertools import chain
from typing import List, Tuple, Iterator

from numpy.core.records import ndarray

from common.imgproc import grid


def from_image(
        image: ndarray,
        sizes: List[Tuple[int,int]],
        steps: List[Tuple[int,int]])->Iterator[Tuple[Tuple[int,int], Tuple[int,int]]]:
    return from_shape(image.shape[:2], sizes, steps)

def from_shape(shape: Tuple[int,int],
        sizes: List[Tuple[int,int]],
        steps: List[Tuple[int,int]])->Iterator[Tuple[Tuple[int,int], Tuple[int,int]]]:
    return chain.from_iterable(
        map(lambda x: grid.from_shape(shape, x[0], x[1]), zip(sizes, steps))
    )