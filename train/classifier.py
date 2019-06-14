from random import shuffle
from typing import Tuple, Dict, List, Generic, TypeVar

import attr
from scipy.stats import randint
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common.config import N_FEATURES_MAX
from train.features.haar import descriptors, computer
import random

T = TypeVar("T")
@attr.s
class RandomFeatureGenerator(Generic[T]):
    features = attr.ib(type=List[T])
    n_max = attr.ib(type=int)

    def rvs(self, random_state: int=10)->List[T]:
        random.seed(random_state)
        shuffle(self.features, random=random.random)
        return self.features[:self.n_max]


def _single_haar_feature(neg: int, pos: int)->Tuple[Pipeline, Dict]:

    haarfs = descriptors()

    pipe = Pipeline([
        ("haar", computer([haarfs[13]])),
        ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=10, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])
    param_dist = {
        "haar__haarf": RandomFeatureGenerator(haarfs, 1),
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"]}
    return pipe, param_dist

def _multi_haar_feature(neg: int, pos: int)->Tuple[Pipeline, Dict]:
    haarfs = descriptors()

    pipe = Pipeline([
        ("haar", computer(haarfs[:N_FEATURES_MAX])),
        ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=10, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])

    param_dist = {
        "haar__haar_features": RandomFeatureGenerator(haarfs, N_FEATURES_MAX),
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"]
    }
    return pipe, param_dist


# load_to_train = _single_haar_feature

load_to_train = _multi_haar_feature

def from_path(fn: str)->Pipeline:
    return joblib.load(fn)

def to_path(fn: str, clf: BaseEstimator)->None:
    joblib.dump(clf, fn)
    return None