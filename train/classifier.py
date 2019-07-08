from train.models.haar.multi.boosting import _boosting_haar_w_selection as HAAR_BOOST_SELECT
from train.models.haar.multi.boosting import _boosting_haar_w_search as HAAR_BOOST
from train.models.hog.svm.kitchen_sinks.svc import _simple_ks_svm as SVM_HOG_KITCHEN
from train.models.hog.svm.sgd import _simple_svm as SVM_KITCHEN

default_load = SVM_HOG_KITCHEN
