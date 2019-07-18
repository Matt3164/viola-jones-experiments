from random import randint

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from pypurr.train.models.utils import LambdaRow, flatten, ProbFromClf


def _extra_trees(neg: int, pos: int):

    sub_pipe = Pipeline([
        ("flattener", LambdaRow(flatten)),
        ("scaler", StandardScaler()),
        ("extra_trees", ExtraTreesClassifier(max_depth=None, n_estimators=50, criterion="entropy", max_features=16, bootstrap=True)),
    ])

    ALPHA = 0.01

    pipe = Pipeline([
        ("xtrapb",
         ProbFromClf(clf=sub_pipe)),
        ("tree",
         DecisionTreeClassifier(criterion="entropy", max_depth=None, class_weight={0: ALPHA / neg, 1: (1.-ALPHA) / pos}))
    ])

    param_dist = {
        "xtrapb__extra_trees__n_estimators": [25, 50, 100],
        "xtrapb__extra_trees__criterion": ["gini", "entropy"],
        "xtrapb__extra_trees__max_depth": [1, 10, None],
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"]
    }

    return pipe, param_dist

    # param_dist = {
    #     "extra_trees__n_estimators": [25, 50, 100],
    #     "extra_trees__criterion": ["gini", "entropy"],
    #     "extra_trees__max_depth": [1, 10, None],
    # }
    #
    # return sub_pipe, param_dist