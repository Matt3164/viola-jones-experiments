from typing import Iterator, Tuple

from numpy.core._multiarray_umath import ndarray, arange

def from_shape(shape: Tuple[int,int],
               size: Tuple[int,int],
               step: Tuple[int,int])->Iterator[Tuple[Tuple[int,int], Tuple[int,int]]]:
    for i in arange(0, shape[0] - size[0], step[0]):
        for j in arange(0, shape[1] - size[1], step[1]):
            yield (i, j), size

# TODO avoid reading from image
def from_image(img: ndarray,
               size: Tuple[int, int],
               step: Tuple[int, int]
               )->Iterator[Tuple[Tuple[int,int], Tuple[int,int]]]:
    return from_shape(img.shape[:2], size, step)

