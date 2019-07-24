from typing import Callable

import numpy as np
from numpy.core._multiarray_umath import ndarray
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import confusion_matrix


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


def flatten(arr: ndarray)->ndarray:
    return arr.flatten()