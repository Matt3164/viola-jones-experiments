from os import makedirs
from os.path import join, exists
from common.config import RUN_PATH, RUN_ID, IM_SIZE
import numpy as np

from train.dataset import load_extracted_data
from train.preprocessing import preprocess


if __name__ == '__main__':

    data = list(load_extracted_data())

    im_size = IM_SIZE

    X = np.empty((len(data), im_size, im_size))
    Y = np.empty((len(data), 1))

    for idx, (impath, label) in enumerate(data):

        image = preprocess(impath)

        X[idx, ::] = image
        Y[idx, 0] = label

    if not exists(join(RUN_PATH, "{:04d}".format(RUN_ID))):
        makedirs(join(RUN_PATH, "{:04d}".format(RUN_ID)))

    np.savez_compressed(
        join(RUN_PATH, "{:04d}".format(RUN_ID), "dataset.npz"), X=X, Y=Y,
    )
