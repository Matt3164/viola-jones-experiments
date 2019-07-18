from typing import List

import attr
from numpy.core.multiarray import ndarray
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Binarizer


@attr.s
class AggStage(TransformerMixin):
    """"""
    clf = attr.ib(type=BaseEstimator)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.clf.predict_proba(X)

    @staticmethod
    def from_clf(clf: BaseEstimator):
        return AggStage(clf=clf)


def from_clfs(clfs: List[BaseEstimator], threshold: float=0.5)->Pipeline:
    return Pipeline([
            ("concatenator", FeatureUnion(list(map(lambda iclf: ("clf_{}".format(iclf[0]), AggStage.from_clf(iclf[1])), enumerate(clfs))))),
            ("Prob1", FunctionTransformer(lambda X: X[:,1::2], validate=False)),
            ("NormL2", FunctionTransformer(lambda X: X.mean(axis=1).reshape(-1,1), validate=False)),
            ("binarizer", Binarizer(threshold=threshold))
        ],
    )

    # return Pipeline([
    #     ("concatenator",
    #      FeatureUnion(list(map(lambda iclf: ("clf_{}".format(iclf[0]), AggStage.from_clf(iclf[1])), enumerate(clfs))))),
    #     ("Prob1", FunctionTransformer(lambda X: X[:, 1::2])),
    #     ("prob_binarizer", Binarizer(threshold=0.5)),
    #     ("NormL2", FunctionTransformer(lambda X: X.mean(axis=1).reshape(-1, 1))),
    #     ("binarizer", Binarizer(threshold=threshold))
    # ],
    # )