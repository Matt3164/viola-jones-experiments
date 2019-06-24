from itertools import chain
from random import shuffle
from typing import List

import attr
import numpy as np

from skimage.feature import haar_like_feature_coord, haar_like_feature
from sklearn.base import BaseEstimator, TransformerMixin

from common.config import NOMINAL_SIZE

HAAR_FEATURE_TYPE = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y','type-4']


class HaarFeatureDescriptor:
    def __init__(self, coord, type):
        """Constructor for HaarFeatureComputer"""
        self.coord = coord
        self.type = type

    def __str__(self) -> str:
        return "Haar feature: {0} {1}".format(self.type, self.coord)


def _haar_coords(feature_type: str):
    coords, types = haar_like_feature_coord(NOMINAL_SIZE, NOMINAL_SIZE, feature_type)
    return zip(coords, types)


def descriptors()->List[HaarFeatureDescriptor]:

    haar_features = list(chain.from_iterable(map(_haar_coords, HAAR_FEATURE_TYPE)))
    shuffle(haar_features)
    haar_features_coords = list(map(lambda x: x[0], haar_features))
    haar_features_types = list(map(lambda x: x[1], haar_features))
    return [HaarFeatureDescriptor(coord=coord, type=type) for coord, type in
              zip(haar_features_coords, haar_features_types)]


@attr.s
class HaarFeatureComputer(BaseEstimator, TransformerMixin):

    haar_features = attr.ib(type=List[HaarFeatureDescriptor])

    def fit(self, X, y=None):
        return self

    def _haar_feat(self, ii):
        return haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1],
            feature_type=np.array([ f.type for f in self.haar_features]),
            feature_coord=np.array([ f.coord for f in self.haar_features]))

    def transform(self, X, y=None):
        return np.array([self._haar_feat(x) for x in X])

def computer(features: List[HaarFeatureDescriptor])->HaarFeatureComputer:
    return HaarFeatureComputer(features)