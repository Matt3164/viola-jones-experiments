from itertools import chain
from random import shuffle
from typing import List

import numpy as np
from skimage.feature import haar_like_feature, haar_like_feature_coord
from sklearn.base import BaseEstimator, TransformerMixin


class HaarFeatureDescriptor:
    def __init__(self, coord, type):
        """Constructor for HaarFeatureComputer"""
        self.coord = coord
        self.type = type

    def __str__(self) -> str:
        return "Haarf: {0} {1}".format(self.type, self.coord)


class HaarFeatureComputer(BaseEstimator, TransformerMixin):
    """"""

    def __init__(self, haarf: HaarFeatureDescriptor) -> None:
        super().__init__()
        self.haarf = haarf

    @property
    def type(self):
        return self.haarf.type

    @property
    def coord(self):
        return self.haarf.coord

    def fit(self, X, y=None):
        return self

    def _haar_feat(self, ii):
        return haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1],
            feature_type=np.array([self.type]),
            feature_coord=np.array([self.coord]))

    def transform(self, X, y=None):
        return np.array([self._haar_feat(x) for x in X])


def haar_coords(feature_type: str):
    coords, types = haar_like_feature_coord(28, 28, feature_type)
    return zip(coords, types)


def haar_features()->List[HaarFeatureDescriptor]:
    features = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y',
                'type-4']
    haar_features = list(chain.from_iterable(map(haar_coords, features)))
    shuffle(haar_features)
    haar_features_coords = list(map(lambda x: x[0], haar_features))
    haar_features_types = list(map(lambda x: x[1], haar_features))
    return [HaarFeatureDescriptor(coord=coord, type=type) for coord, type in
              zip(haar_features_coords, haar_features_types)]