from os.path import join
from typing import Tuple, Dict

from scipy.stats import randint
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common.config import RUN_PATH
from cli.filter_negatives import run_id

from train import haarfs, neg, pos
from train.features import HaarFeatureComputer


def load_to_train()->Tuple[Pipeline, Dict]:
    pipe = Pipeline([
        ("haar", HaarFeatureComputer(haarfs[13])),
        ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=10, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])
    param_dist = {
        "haar__haarf": haarfs,
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"]}
    return pipe, param_dist


def load_trained(run_id: int)->Pipeline:
    clf_path = join(RUN_PATH, "{:04d}".format(run_id), "classifier.pickle")
    return joblib.load(clf_path)