from os import makedirs
from common.config import RUN_ID, NOMINAL_SIZE, POSITIVE_PATH
import numpy as np
from common.iotools.dataset import to_path, from_paths
from common.iotools.labeled_path import from_folders
from train.path_utils import run, dataset, negatives
from train.preprocessing.window import from_path


if __name__ == '__main__':

    data = list(from_folders([negatives(RUN_ID), POSITIVE_PATH]))

    X, Y = from_paths(data)

    makedirs(run(RUN_ID), exist_ok=True)

    to_path(dataset(RUN_ID), X, Y)
