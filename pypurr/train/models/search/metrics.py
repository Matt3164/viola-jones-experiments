from sklearn.metrics import make_scorer, recall_score, accuracy_score, fbeta_score, f1_score


def mix(ytrue, ypred):
    return 0.5*recall_score(ytrue, ypred)+0.5*accuracy_score(ytrue, ypred)


def fbeta(ytrue, ypred):
    return fbeta_score(ytrue, ypred, beta=1)


metrics = {
    "recall": make_scorer(recall_score),
    "acc": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score),
    "fbeta": make_scorer(fbeta),
    "mix": make_scorer(mix)
}

