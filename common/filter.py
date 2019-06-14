from typing import List, Tuple

from numpy import vstack
from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator

def by_clf(X: ndarray, Xmetadata: List, classifier: BaseEstimator)->Tuple[ndarray, List, ndarray, List]:
    preds = classifier.predict(X)

    Xpos = X[preds == True, ::]

    Xmetapos = list(
        map(lambda x: x[1], filter(lambda x: x[0], zip(preds, Xmetadata))))

    Xneg = X[preds == False, ::]

    Xmetaneg = list(
        map(lambda x: x[1], filter(lambda x: not x[0], zip(preds, Xmetadata))))

    return Xpos, Xmetapos, Xneg, Xmetaneg

def by_cascade(X: ndarray, Xmetadata: List, classifiers: List[BaseEstimator])->Tuple[ndarray, List, ndarray, List]:

    kept_array = X

    kept_metadata = Xmetadata

    filtered_array = []
    filtered_metadata = []

    for clf in classifiers:

        kept_array, kept_metadata, _filtered_array, _filtered_metadata = by_clf(kept_array, kept_metadata, clf)

        filtered_array.append(_filtered_array)
        filtered_metadata.extend(_filtered_metadata)

    return kept_array, kept_metadata, vstack(filtered_array), filtered_metadata

