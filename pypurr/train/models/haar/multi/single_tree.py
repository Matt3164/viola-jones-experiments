from typing import Tuple, Dict

from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from pypurr.common.config import N_FEATURES_MAX
from pypurr.train.features.haar import descriptors, computer
from pypurr.train.models.utils import RandomFeatureGenerator


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