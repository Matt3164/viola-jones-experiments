from glob import glob
from itertools import chain
from os import listdir
from os.path import join
from typing import Tuple

import numpy as np
from pandas import DataFrame

from common.config import DATA_PATH
from train.path_utils import image_df


def _prepare()->None:

    data_path = DATA_PATH

    image_files = chain.from_iterable(
        map(lambda folder: glob(join(data_path, folder, "*.jpg")), listdir(data_path))
    )

    image_files_1 = chain.from_iterable(
        map(lambda folder: glob(join(data_path, folder, "*.jpg")), listdir(data_path))
    )

    df = DataFrame.from_records(map(_flatten, zip(image_files_1, map(_bbox_from_filepath, image_files))),
                                columns=["ImagePath", "Xmin", "Xmax", "Ymin", "Ymax"])

    df.to_csv(image_df())

    return None


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


def get()->None:
    """
    Should download dataset files

    https://www.kaggle.com/crawford/cat-dataset
    https://archive.org/details/CAT_DATASET
    https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip
    https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip

    Returns:

    """
    return None