from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from common.config import RUN_ID, RANDOM_SEARCH
from train.classifier import load_to_train
from train.models.search.search import from_model_and_params
from train.models.utils import to_path
from common.iotools.dataset import from_path
from train.path_utils import dataset, classifier

if __name__ == '__main__':

    dry_run = False

    X, Y = from_path(dataset(RUN_ID))

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8, test_size=0.2)

    pos = ytrain.sum()

    neg = ytrain.shape[0] - pos

    clf, param_dist = load_to_train(neg, pos)

    if RANDOM_SEARCH:
        clf = from_model_and_params(clf, param_dist)

    clf.fit(xtrain, ytrain.ravel())

    for tag, x, y in [
        ("train", xtrain, ytrain), ("test", xtest, ytest)]:

        print("--- Fold {} ----".format(tag))

        ypred = clf.predict(x)

        for val_metric in [confusion_matrix, accuracy_score]:

            print("{0}  --> {1}".format(val_metric.__name__, val_metric(y, clf.predict(x))))


    if not dry_run:
        to_path(classifier(RUN_ID), clf )
