from typing import Tuple, List

import numpy as np

from pypurr.common.config import NOMINAL_SIZE


def from_path(fn:str)->Tuple[np.ndarray,np.ndarray]:
    dataset = np.load(fn)
    return dataset["X"], dataset["Y"]


def to_path(fn : str, X: np.ndarray, Y: np.ndarray)-> None:
    np.savez_compressed(
        fn, X=X, Y=Y,
    )

    return None


def from_paths(paths: List[Tuple[str, int]])->Tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(paths), NOMINAL_SIZE, NOMINAL_SIZE))
    Y = np.empty((len(paths), 1))

    for idx, (impath, label) in enumerate(paths):

        image = from_path(impath)

        X[idx, ::] = image
        Y[idx, 0] = label

    return X, Y