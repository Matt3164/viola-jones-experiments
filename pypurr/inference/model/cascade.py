from typing import List, Tuple

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

from pypurr.inference.model.cascade_stage import CascadeStage
from pypurr.train.models.utils import LambdaRow


def _to_step(idxclf_tup: Tuple[int, BaseEstimator])->Tuple[str, BaseEstimator]:

    idx, clf = idxclf_tup
    return ("classifier_{}".format(idx), CascadeStage.from_clf(clf=clf))

def from_clfs(clfs: List[BaseEstimator])->Pipeline:

    return Pipeline(
        list(map(_to_step, enumerate(clfs))) + [("summer", LambdaRow(row_func=lambda x: x.sum())), ("binarizer", Binarizer())]
    )
