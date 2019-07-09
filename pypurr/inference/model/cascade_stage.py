import attr
from numpy.core.multiarray import ndarray
from sklearn.base import TransformerMixin, BaseEstimator


@attr.s
class CascadeStage(TransformerMixin):
    """"""
    clf = attr.ib(type=BaseEstimator)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        bool_idx = X.sum(axis=(1,2))

        selected_X = X[bool_idx > 0, ::]

        if selected_X.shape[0]>0:

            pred = self.clf.predict(selected_X)

            selected_X[pred==0,::] = 0.

            X[bool_idx > 0, ::] = selected_X

            return X
        else:
            return X

    @staticmethod
    def from_clf(clf: BaseEstimator):
        return CascadeStage(clf=clf)
