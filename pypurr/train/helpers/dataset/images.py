from os.path import join, exists
from typing import Iterator
from numpy.core.multiarray import ndarray

from pypurr.common.helpers import image

def from_path(path: str)->Iterator[ndarray]:
    return map(
        lambda x: image.from_path(x),
        image.find_pngs(path)
    )

def to_path(path: str, images: Iterator[ndarray])->bool:

    for idx, img in enumerate(images):

        fn = join(path, "im_{:05d}.png".format(idx))

        if not exists(fn):
            image.to_path(fn, img)

    return True