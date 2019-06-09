from glob import glob
from itertools import chain, repeat
from os import listdir
from os.path import join
from typing import Tuple, Iterator, List

import numpy as np
from pandas import read_csv, DataFrame

from common.config import WORKSPACE, POSITIVE_PATH, NEGATIVE_PATH, RUN_ID, RUN_PATH


def load(shuffle: bool=False, sample: bool=False, n_samples: int=100)->Iterator[Tuple[str, Tuple[int,int,int,int]]]:

    dataframe_path = join(WORKSPACE, "images.csv")

    df = read_csv(dataframe_path, index_col=0)

    if shuffle:
        df = df.sample(frac=1.)

    if sample:
        df = df.sample(n=n_samples)

    assert isinstance(df, DataFrame)

    for idx, elmt in df.iterrows():

        yield elmt.ImagePath, (elmt.Xmin, elmt.Xmax, elmt.Ymin, elmt.Ymax)


def prepare()->None:

    data_path = WORKSPACE

    image_files = chain.from_iterable(
        map(lambda folder: glob(join(data_path, folder, "*.jpg")), listdir(data_path))
    )

    image_files_1 = chain.from_iterable(
        map(lambda folder: glob(join(data_path, folder, "*.jpg")), listdir(data_path))
    )

    df = DataFrame.from_records(map(_flatten, zip(image_files_1, map(_bbox_from_filepath, image_files))),
                                columns=["ImagePath", "Xmin", "Xmax", "Ymin", "Ymax"])

    df.to_csv(join(data_path, "images.csv"))

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

    Returns:

    """

    # wget $URL
    return None


def positives()->List[str]:
    return glob(join(POSITIVE_PATH, "*.png"))


def negatives()->List[str]:
    return glob(join(NEGATIVE_PATH, "{:04d}".format(RUN_ID), "*.png"))


def load_extracted_data()->Iterator[Tuple[str, int]]:
    """


    Returns:
        Data iterator: path, label

    """
    pos_files = positives()
    neg_files = negatives()
    return chain.from_iterable(
        [zip(pos_files, repeat(1, len(pos_files))), zip(neg_files, repeat(0, len(neg_files)))]
    )


def load_xy()->Tuple[np.ndarray,np.ndarray]:
    dataset = np.load(join(RUN_PATH, "{:04d}".format(RUN_ID), "dataset.npz"))
    return dataset["X"], dataset["Y"]