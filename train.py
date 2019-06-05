from itertools import chain
from os.path import join
from random import shuffle

import numpy as np
from scipy.stats import randint
from skimage.feature import haar_like_feature, haar_like_feature_coord
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, recall_score, make_scorer, accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import RUN_PATH, RUN_ID


class HaarFeatureDescriptor:
    def __init__(self, coord, type):
        """Constructor for HaarFeatureComputer"""
        self.coord = coord
        self.type = type

    def __str__(self) -> str:
        return "Haarf: {0} {1}".format(self.type, self.coord)


class HaarFeatureComputer(BaseEstimator, TransformerMixin):
    """"""

    def __init__(self, haarf: HaarFeatureDescriptor) -> None:
        super().__init__()
        self.haarf = haarf

    @property
    def type(self):
        return self.haarf.type

    @property
    def coord(self):
        return self.haarf.coord

    def fit(self, X, y=None):
        return self

    def _haar_feat(self, ii):
        return haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1],
            feature_type=np.array([self.type]),
            feature_coord=np.array([self.coord]))

    def transform(self, X, y=None):
        return np.array([self._haar_feat(x) for x in X])

def haar_coords(feature_type: str):
    coords, types = haar_like_feature_coord(28, 28, feature_type)
    return zip(coords, types)


def mix(ytrue, ypred):
    return 0.5*recall_score(ytrue, ypred)+0.5*accuracy_score(ytrue, ypred)

def fbeta(ytrue, ypred):
    return fbeta_score(ytrue, ypred, beta=1)


if __name__ == '__main__':

    dry_run = False

    dataset = np.load(join(RUN_PATH, "{:04d}".format(RUN_ID), "dataset.npz"))

    X = dataset["X"]
    Y = dataset["Y"]

    features = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y',
                'type-4']

    haar_features = list(chain.from_iterable(map(haar_coords, features)))

    shuffle(haar_features)

    # max_haar_features = 5000
    #
    # haar_features = haar_features[:max_haar_features]

    haar_features_coords = list(map(lambda x: x[0], haar_features))

    haar_features_types = list(map(lambda x: x[1], haar_features))

    haarfs = [HaarFeatureDescriptor(coord=coord, type=type) for coord, type in
              zip(haar_features_coords, haar_features_types)]

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8, test_size=0.2)

    pos = ytrain.sum()

    neg = ytrain.shape[0] - pos


    pipe = Pipeline([
        ("haar", HaarFeatureComputer(haarfs[13])),
        ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=10, class_weight={0: 0.01 / neg, 1: 0.99 / pos}))
    ])


    param_dist = {
        "haar__haarf": haarfs,
        "tree__max_depth": [1, 2, 5, 10, None],
      "tree__min_samples_split": randint(2, 11),
      "tree__criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 250

    clf = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=2, iid=False, verbose=2, scoring={
            "recall":make_scorer(recall_score),
            "acc": make_scorer(accuracy_score),
            "fbeta": make_scorer(fbeta),
            "mix": make_scorer(mix)
        },
         refit="fbeta"
         )



    clf.fit(xtrain, ytrain)

    print(clf.score(xtrain, ytrain))
    print(clf.score(xtest, ytest))

    print(confusion_matrix(ytrain, clf.predict(xtrain)))

    print(confusion_matrix(ytest, clf.predict(xtest)))

    if not dry_run:
        joblib.dump(clf, join(RUN_PATH, "{:04d}".format(RUN_ID), "classifier.pickle"))
