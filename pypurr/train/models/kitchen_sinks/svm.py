from random import randint
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from pypurr.train.models.search.search import from_model_and_params
from pypurr.train.models.utils import LambdaRow, ProbFromClf, flatten


def _simple_svm(neg: int, pos: int):

    sub_pipe = Pipeline([
        ("flattener", LambdaRow(flatten)),
        ("kitchen_sinks", RBFSampler(n_components=250)),
        ("svm", SGDClassifier(loss="hinge", penalty="l2", alpha=0.001, max_iter=25, tol=1e-3, learning_rate="optimal")),
    ])

    sub_param_dist = {
        "svm__loss": ["hinge", "log", "modified_huber"],
        "svm__penalty": ["l1", "l2", "elasticnet"],
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
