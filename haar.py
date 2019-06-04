from random import shuffle

import cv2
from skimage.data import chelsea
from skimage.feature.haar import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
import numpy as np

if __name__ == '__main__':

    image = chelsea()

    image = cv2.resize(image, (28, 28))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ii = integral_image(image)
    #
    features = np.array(['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y',
                         'type-4'])

    print(
        haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1],
            feature_type=features
        ).shape)

    print(
        haar_like_feature_coord(28, 28, 'type-2-x')
    )

    coords, types = haar_like_feature_coord(28, 28, 'type-2-x')

    print(
        haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1],
            feature_type=types,
            feature_coord=coords
        ).shape)

    # feature_type = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y',
    #                      'type-4']
    #
    # img = np.ones((5, 5), dtype=np.int8)
    #
    # img_ii = integral_image(img)

    # if isinstance(feature_type, list):
    #     # shuffle the index of the feature to be sure that we are output
    #     # the features in the same order
    #     shuffle(feature_type)
    #     feat_coord, feat_type = zip(*[haar_like_feature_coord(5, 5, feat_t)
    #                                   for feat_t in feature_type])
    #     feat_coord = np.concatenate(feat_coord)
    #     feat_type = np.concatenate(feat_type)
    # else:

    # feat_coord, feat_type = haar_like_feature_coord(5, 5, feature_type[0])
    #
    # haar_feature_precomputed = haar_like_feature(img_ii, 0, 0, 5, 5,
    #                                              feature_type=feat_type,
    #                                              feature_coord=feat_coord)

    """
    features = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y',
                         'type-4']

    haar_features = list(chain.from_iterable(map(haar_coords, features)))

    shuffle(haar_features)

    haar_features = haar_features[:10]

    haar_features_coords = list(map(lambda x: x[0], haar_features))

    haar_features_types = list(map(lambda x: x[1], haar_features))

    image = chelsea()

    image = cv2.resize(image, (28, 28))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ii = integral_image(image)

    print(
        haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1],
            feature_type=np.array(haar_features_types),
            feature_coord=np.array(haar_features_coords)
        ).shape)"""
