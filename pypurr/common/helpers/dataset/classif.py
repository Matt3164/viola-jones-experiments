import numpy as np
from itertools import chain, repeat
from typing import List, Iterator, Tuple, Callable

from pypurr.common.helpers import image


def from_path(fn:str)->Tuple[np.ndarray,np.ndarray]:
    dataset = np.load(fn)
    return dataset["X"], dataset["Y"]


def to_path(fn : str, X: np.ndarray, Y: np.ndarray)-> None:
    np.savez_compressed(
        fn, X=X, Y=Y,
    )

    return None


def from_iterable(
        paths: List[Tuple[np.ndarray, int]],
        target_size: int=32
    )->Tuple[np.ndarray, np.ndarray]:


    X = np.empty((len(paths), target_size, target_size))
    Y = np.empty((len(paths), 1))

    for idx, (img, label) in enumerate(paths):

        X[idx, ::] = img
        Y[idx, 0] = label

    return X, Y


def from_folders(
        folders: List[str],
        preproc_fn: Callable[[np.ndarray], np.ndarray]
    )->Iterator[Tuple[np.ndarray, int]]:
    """


    Returns:
        Data iterator: path, label

    """
    return chain.from_iterable(
        map(lambda elmt: zip(preproc_fn(image.from_path(elmt[1])), repeat(elmt[0])), enumerate(map(image.find_pngs, folders)))
    )


