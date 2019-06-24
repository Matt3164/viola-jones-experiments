from random import shuffle
from typing import Tuple, Dict

from scipy.stats import randint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common.config import N_FEATURES_MAX
from train.features.haar import descriptors, computer
from train.models.search.search import from_model_and_params
from train.models.utils import ProbFromClf, RandomFeatureGenerator

def _boosting_haar_w_selection(neg: int, pos: int)->Tuple[Pipeline, Dict]:

    haarfs = descriptors()

    shuffle(haarfs)

    pipe = Pipeline([
        ("haar", computer(haarfs[:10*N_FEATURES_MAX])),
        ("feat_sel", SelectKBest(f_classif, k=N_FEATURES_MAX)),
        ("boosting", ProbFromClf(clf=GradientBoostingClassifier(max_depth=3, subsample=0.1, n_estimators=N_FEATURES_MAX))),
        ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=3, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])

    return pipe, dict()

def _boosting_haar_w_search(neg: int, pos: int)->Tuple[Pipeline, Dict]:
    haarfs = descriptors()

    shuffle(haarfs)

    pipe = Pipeline([
        ("haar", computer(haarfs[:N_FEATURES_MAX])),
        ("boosting", ProbFromClf(clf=GradientBoostingClassifier(max_depth=3, subsample=0.1, n_estimators=N_FEATURES_MAX))),
        ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=3, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])

    param_dist = {
        "haar__haar_features": RandomFeatureGenerator(haarfs, N_FEATURES_MAX),
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"]
    }

    return pipe, param_dist


def _feature_selection_w_boosting(neg: int, pos: int)->Tuple[Pipeline, Dict]:
    haarfs = descriptors()

    shuffle(haarfs)

    sub_pipe = Pipeline([
        ("haar", computer(haarfs[:N_FEATURES_MAX])),
        ("boosting", GradientBoostingClassifier(max_depth=3, subsample=0.1, n_estimators=N_FEATURES_MAX)),
    ])

    sub_param_dist = {
        "haar__haar_features": RandomFeatureGenerator(haarfs, N_FEATURES_MAX),
        "boosting__max_depth": [1,3,5,10],
        "boosting__subsample": [0.1, 0.5, None],
        "boosting__n_estimators": [10, 25, 100, N_FEATURES_MAX, 2*N_FEATURES_MAX],
    }

    pipe = Pipeline([
        ("boosting",
         ProbFromClf(clf=from_model_and_params(sub_pipe, sub_param_dist))),
        ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=3, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])

    param_dist = {
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"]
    }

    return pipe, param_dist
