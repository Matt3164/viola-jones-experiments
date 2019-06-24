from scipy.stats import randint
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from train.models.hog.computer import HOG
from train.models.search.search import from_model_and_params
from train.models.utils import LambdaRow, ProbFromClf

def _simple_ks_svm(neg: int, pos: int):

    sub_pipe = Pipeline([
        ("hog", LambdaRow(HOG)),
        ("kitchen_sinks", RBFSampler(n_components=250)),
        ("svm", SVC(C=1.0, kernel="linear", tol=1e-3, probability=True)),
    ])

    pipe = Pipeline([
        ("svmpb",
         ProbFromClf(clf=sub_pipe)),
        ("tree",
         DecisionTreeClassifier(criterion="entropy", max_depth=3, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])

    param_dist = {
        "tree__max_depth": [1, 2, 5, 10, None],
        "tree__min_samples_split": randint(2, 11),
        "tree__criterion": ["gini", "entropy"],
        "svmpb__svm__C": [1.0, 10.0, 1e-1],
        "svmpb__svm__kernel": ['linear'],
    }

    return pipe, param_dist
