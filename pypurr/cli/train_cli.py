from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from pypurr.common.config import RUN_ID, RANDOM_SEARCH
from pypurr.train.classifier import default_load
from pypurr.train.models.search.search import from_model_and_params
from pypurr.common.helpers.model import to_path
from pypurr.deprecated.iotools import from_path
from pypurr.train.path_utils import dataset, classifier


def _train(run_id: int):

    X, Y = from_path(dataset(run_id))
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8, test_size=0.2)
    pos = ytrain.sum()
    neg = ytrain.shape[0] - pos
    clf, param_dist = default_load(neg, pos)
    if RANDOM_SEARCH:
        clf = from_model_and_params(clf, param_dist)
    clf.fit(xtrain, ytrain.ravel())
    for tag, x, y in [
        ("train", xtrain, ytrain), ("test", xtest, ytest)]:

        print("--- Fold {} ----".format(tag))

        for val_metric in [confusion_matrix, accuracy_score]:

            print("{0}  --> {1}".format(val_metric.__name__, val_metric(y, clf.predict(x))))

    to_path(classifier(run_id), clf)


if __name__ == '__main__':

    _train(RUN_ID)
