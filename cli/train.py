from os.path import join

from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, recall_score, make_scorer, accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from common.config import RUN_PATH, RUN_ID
from train.classifier import load_to_train
from train.dataset import load_xy


def mix(ytrue, ypred):
    return 0.5*recall_score(ytrue, ypred)+0.5*accuracy_score(ytrue, ypred)

def fbeta(ytrue, ypred):
    return fbeta_score(ytrue, ypred, beta=1)


if __name__ == '__main__':

    dry_run = False

    X, Y = load_xy()

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8, test_size=0.2)

    pos = ytrain.sum()

    neg = ytrain.shape[0] - pos

    pipe, param_dist = load_to_train()

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
