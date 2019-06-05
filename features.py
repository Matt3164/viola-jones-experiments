from glob import glob
from itertools import repeat, chain
from os import makedirs
from os.path import join, exists
from random import shuffle

import cv2
from skimage.data import chelsea
from skimage.feature import haar_like_feature_coord, haar_like_feature
from skimage.transform import integral_image

from config import POSITIVE_PATH, NEGATIVE_PATH, RUN_PATH, RUN_ID
import numpy as np




def haar_coords(feature_type: str):
    coords, types = haar_like_feature_coord(im_size, im_size, feature_type)
    return zip(coords, types)


if __name__ == '__main__':

    pos_files = glob(join(POSITIVE_PATH, "*.png"))
    neg_files = glob(join(NEGATIVE_PATH, "{:04d}".format(RUN_ID), "*.png"))

    pathlabel_iter = chain.from_iterable(
        [zip(pos_files, repeat(1, len(pos_files))), zip(neg_files, repeat(0, len(neg_files)))]
    )

    im_size = 28

    # features = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y',
    #             'type-4']
    #
    # haar_features = list(chain.from_iterable(map(haar_coords, features)))
    #
    # shuffle(haar_features)
    #
    # max_haar_features = 2500
    #
    # haar_features = haar_features[:max_haar_features]
    #
    # haar_features_coords = list(map(lambda x: x[0], haar_features))
    #
    # haar_features_types = list(map(lambda x: x[1], haar_features))
    #
    # X = np.empty((len(pos_files) + len(neg_files), max_haar_features))

    X = np.empty((len(pos_files) + len(neg_files), im_size, im_size))
    Y = np.empty((len(pos_files) + len(neg_files), 1))

    for idx, (impath, label) in enumerate(pathlabel_iter):

        image = cv2.imread(impath)

        image = cv2.resize(image, (im_size, im_size))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = integral_image(image)

        #
        # feats = haar_like_feature(
        #     ii, 0, 0, ii.shape[0], ii.shape[1],
        #     feature_type=np.array(haar_features_types),
        #     feature_coord=np.array(haar_features_coords)
        # )

        X[idx, ::] = image
        Y[idx, 0] = label

    if not exists(join(RUN_PATH, "{:04d}".format(RUN_ID))):
        makedirs(join(RUN_PATH, "{:04d}".format(RUN_ID)))

    np.savez_compressed(
        join(RUN_PATH, "{:04d}".format(RUN_ID), "dataset.npz"), X=X, Y=Y,
        # features_coords=np.array(haar_features_coords), features_types=np.array(haar_features_types)
    )
