from typing import Tuple

import numpy as np

def from_path(fn:str)->Tuple[np.ndarray,np.ndarray]:
    dataset = np.load(fn)
    return dataset["X"], dataset["Y"]

def to_path(fn : str, X: np.ndarray, Y: np.ndarray)-> None:
    np.savez_compressed(
        fn, X=X, Y=Y,
    )

    return None