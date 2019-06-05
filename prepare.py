from glob import glob
from itertools import chain
from os import listdir
from os.path import join
import numpy as np
from typing import Tuple

from pandas import DataFrame

from config import WORKSPACE


def _bbox_from_filepath(image_file: str) -> Tuple[int, int, int, int]:
    with open(image_file + ".cat", "r") as f:
        content = f.read()

    pos = [int(e) for e in content.split(" ") if e != '']

    pos = np.array(pos)[1:]

    x = pos[::2]

    y = pos[1::2]

    return x.min(), x.max(), y.min(), y.max()


def _flatten(s: Tuple[str, Tuple[int, int, int, int]]) -> Tuple[str, int, int, int, int]:
    im, bbox = s
    return (im, bbox[0], bbox[1], bbox[2], bbox[3])


if __name__ == '__main__':

    data_path = WORKSPACE

    workspace = WORKSPACE

    image_files = chain.from_iterable(
        map(lambda folder: glob(join(data_path, folder, "*.jpg")), listdir(data_path))
    )

    image_files_1 = chain.from_iterable(
        map(lambda folder: glob(join(data_path, folder, "*.jpg")), listdir(data_path))
    )

    df = DataFrame.from_records(map(_flatten, zip(image_files_1, map(_bbox_from_filepath, image_files))),
                                columns=["ImagePath", "Xmin", "Xmax", "Ymin", "Ymax"])

    df.to_csv(join(workspace, "./images.csv"))
