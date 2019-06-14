from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from common.config import RUN_ID
from train.classifier import load_to_train, to_path
from train.datasets.array import from_path
from train.search_metrics import metrics, optimized
from train.path_utils import dataset, classifier

if __name__ == '__main__':

    dry_run = False

    X, Y = from_path(dataset(RUN_ID))

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8, test_size=0.2)

    pos = ytrain.sum()

    neg = ytrain.shape[0] - pos

    pipe, param_dist = load_to_train(neg, pos)

    # run randomized search
    n_iter_search = 25

    clf = RandomizedSearchCV(pipe, param_distributions=param_dist,
                             n_iter=n_iter_search, cv=2, iid=False, verbose=2, scoring=metrics,
                             refit=optimized,
                             n_jobs=4
                             )

    clf.fit(xtrain, ytrain)


    for tag, x, y in [
        ("train", xtrain, ytrain), ("test", xtest, ytest)]:

        print("--- Fold {} ----".format(tag))

        ypred = clf.predict(x)

        for val_metric in [confusion_matrix, accuracy_score]:

            print("{0}  --> {1}".format(val_metric.__name__, val_metric(y, clf.predict(x))))


    if not dry_run:
        to_path(classifier(RUN_ID), clf )