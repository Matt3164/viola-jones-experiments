import random
from random import shuffle
from typing import TypeVar, Generic, List, Callable

import attr
from numpy.core.multiarray import ndarray
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np

T = TypeVar("T")


@attr.s
class RandomFeatureGenerator(Generic[T]):
    features = attr.ib(type=List[T])
    n_max = attr.ib(type=int)

    def rvs(self, random_state: int=10)->List[T]:
        random.seed(random_state)
        shuffle(self.features, random=random.random)
        return self.features[:self.n_max]


class ProbFromClf(TransformerMixin, BaseEstimator):
    """"""

    def __init__(self, clf: BaseEstimator):
        """Constructor for ProbFromClf"""
        self.clf = clf

    def fit(self, X, y=None):
        self.clf.fit(X,y)
        print("Sub clf: ---> {}".format(confusion_matrix(y, self.clf.predict(X))))
        return self

    def transform(self, X):
        return self.clf.predict_proba(X)

    def get_params(self, deep=True):
        return dict(clf=self.clf)

    def set_params(self, **params):
        return self.clf.set_params(**params)


def from_path(fn: str)->Pipeline:
    return joblib.load(fn)


def to_path(fn: str, clf: BaseEstimator)->None:
    joblib.dump(clf, fn)
    return None


class LambdaRow(TransformerMixin):
    """"""

    def __init__(self,
                 row_func: Callable[[ndarray], ndarray]):
        """Constructor for LambdaRow"""
        self.func = row_func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _arr = np.array([self.func(x) for x in X])
        if len(_arr.shape)>=2:
            return _arr
        else:
            return _arr.reshape(-1,1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)


