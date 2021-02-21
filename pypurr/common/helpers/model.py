import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def from_path(fn: str) -> Pipeline:
    return joblib.load(fn)


def to_path(fn: str, clf: BaseEstimator) -> None:
    joblib.dump(clf, fn)
    return None
