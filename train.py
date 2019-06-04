from os.path import join
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import RUN_PATH, RUN_ID

if __name__ == '__main__':

    dry_run = False

    dataset = np.load(join(RUN_PATH, "{:04d}".format(RUN_ID), "dataset.npz"))

    X = dataset["X"]
    Y = dataset["Y"]

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.5, test_size=0.5)

    pos = ytrain.sum()

    neg = ytrain.shape[0] - pos

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, class_weight={0: 0.01 / neg, 1: 0.99 / pos})

    clf.fit(xtrain, ytrain)

    print(clf.score(xtrain, ytrain))
    print(clf.score(xtest, ytest))

    print(confusion_matrix(ytrain, clf.predict(xtrain)))

    print(confusion_matrix(ytest, clf.predict(xtest)))

    if not dry_run:
        joblib.dump(clf, join(RUN_PATH, "{:04d}".format(RUN_ID), "classifier.pickle"))
