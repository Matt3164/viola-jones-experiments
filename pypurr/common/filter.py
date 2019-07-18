from typing import List, Tuple
from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator

def by_clf(X: ndarray, Xmetadata: List, classifier: BaseEstimator)->Tuple[ndarray, List, ndarray, List]:
    preds = classifier.transform(X)

    Xmetapos = map(lambda x: x[1], filter(lambda x: x[0], zip(preds, Xmetadata)))
    Xmetaneg = map(lambda x: x[1], filter(lambda x: ~x[0], zip(preds, Xmetadata)))

    return X[preds.astype(bool).flatten(), ::], Xmetapos, X[~preds.astype(bool).flatten(), ::], Xmetaneg