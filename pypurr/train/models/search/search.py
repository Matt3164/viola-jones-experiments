from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from pypurr.train.models.search.metrics import metrics
from pypurr.train.models.search.settings import N_ITER, OPTIMIZED, N_JOBS


def from_model_and_params(model: BaseEstimator, param_dist: Dict)->BaseEstimator:

    return RandomizedSearchCV(
        model, param_distributions=param_dist,
                             n_iter=N_ITER, cv=2, iid=False, verbose=2, scoring=metrics,
                             refit=OPTIMIZED,
                             n_jobs=N_JOBS
                             )
