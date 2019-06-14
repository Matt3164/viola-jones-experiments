from os import makedirs
from common.config import RUN_ID, IM_SIZE, POSITIVE_PATH
import numpy as np
from train.datasets.array import to_path
from train.datasets.image import from_folders
from train.path_utils import run, dataset, negatives
from train.preprocessing import from_path


if __name__ == '__main__':

    data = list(from_folders([negatives(RUN_ID), POSITIVE_PATH]))

    X = np.empty((len(data), IM_SIZE, IM_SIZE))
    Y = np.empty((len(data), 1))

    for idx, (impath, label) in enumerate(data):

        image = from_path(impath)

        X[idx, ::] = image
        Y[idx, 0] = label


    makedirs(run(RUN_ID), exist_ok=True)

    to_path(dataset(RUN_ID), X, Y)
