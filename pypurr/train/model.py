from random import randint

from skimage.transform import integral_image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from pypurr.train.helpers.model import ProbFromClf, LambdaRow, flatten


def build(neg: int, pos: int):

    sub_pipe = Pipeline([
        ("integral_image", LambdaRow(integral_image)),
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