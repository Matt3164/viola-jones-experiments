from train.models.haar.multi.boosting import _boosting_haar_w_selection

# load_to_train = _boosting_haar_w_selection
from train.models.hog.svm.kitchen_sinks.svc import _simple_ks_svm
from train.models.hog.svm.sgd import _simple_svm

load_to_train = _simple_ks_svm
