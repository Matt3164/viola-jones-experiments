from itertools import repeat, tee, islice, chain
from typing import Iterator, Tuple

from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from pypurr.common.iotools.dataset import from_path
from pypurr.train.path_utils import dataset
from skimage.util import view_as_windows
import numpy as np

"""

Extract small windows 

"""

def _windows(arr: ndarray, label: int)->Iterator[Tuple[ndarray, int]]:
    return zip(view_as_windows(arr, (7, 7), step=7).reshape(-1,7,7), repeat(label))

def _predict_with_subclf(arr: ndarray, clf: BaseEstimator)->ndarray:

    windows = view_as_windows(arr, (7,7), step=7).reshape(-1,7*7)

    predictions = clf.predict(windows)




if __name__ == '__main__':

    X, Y = from_path(dataset(4))

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8, test_size=0.2)

    print(xtrain.shape)

    new_samples = chain.from_iterable(map(lambda elmt :_windows(elmt[0], elmt[1]), zip(xtrain, ytrain)))

    x_sample, y_sample = tee(new_samples, 2)

    n_max_samples = 5000

    x_sample, y_sample = islice(map(lambda x: x[0], x_sample), n_max_samples), islice(map(lambda x: x[1], y_sample), n_max_samples)

    X, Y = np.array(list(x_sample)), np.array(list(y_sample))

    print(np.unique(Y, return_counts=True))

    rf = RandomForestClassifier(max_depth=None, max_features="log2", n_estimators=100)

    n_samples, _, _ = X.shape

    X = X.reshape(n_samples, -1)

    rf.fit(X, Y.reshape(-1,1))

    print(rf.score(X, Y))

    print(confusion_matrix(Y.reshape(-1,1), rf.predict(X)))

