from random import randint
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from train.models.hog.computer import HOG
from train.models.search.search import from_model_and_params
from train.models.utils import LambdaRow, ProbFromClf


def _simple_svc(neg: int, pos: int):

    sub_pipe = Pipeline([
        ("hog", LambdaRow(HOG)),
        ("svm", SVC(C=1.0, kernel="rbf", tol=1e-3)),
    ])

    sub_param_dist = {
        "svm__C": [1.0, 10.0, 1e-1],
        "svm__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    }

    pipe = Pipeline([
        ("svmpb",
         ProbFromClf(clf=from_model_and_params(sub_pipe, sub_param_dist))),
        ("tree",
         DecisionTreeClassifier(criterion="entropy", max_depth=3, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])

    param_dist = {
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"]
    }

    return pipe, param_dist
