from itertools import chain, islice, tee
from os import makedirs
from typing import List, Tuple, Iterator

import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator

from pypurr.common.config import MAX_NEGATIVE_EXAMPLES, IOU_THRESHOLD, BATCH_SIZE, RUN_ID, THRESHOLD, SCALES
from pypurr.common.helpers.model import from_path as clf_from_path
from pypurr.common.preprocessing.window import from_array
from pypurr.deprecated.iotools.images import to_path as imgs_to_path
from pypurr.train import scan_iou
from pypurr.train.helpers import dataset
from pypurr.train.path_utils import negatives, classifier, image_df


def _filter_w_metadata(data: List[Tuple[ndarray, ndarray]],
                       classifiers: List[BaseEstimator]
                       ) -> Iterator[Tuple[ndarray, ndarray]]:
    X = np.array(list(map(lambda x: x[0], data)))
    keptX, keptMeta, _, _ = by_clf(X, list(map(lambda x: x[1], data)),
                                   aggregation.from_clfs(classifiers, threshold=THRESHOLD))
    return zip(keptX, keptMeta)


def _extract_negatives(run_id: int):
    negative_storage_path = negatives(run_id)
    classifiers = [clf_from_path(classifier(_run_id)) for _run_id in np.arange(0, run_id)]
    makedirs(negative_storage_path, exist_ok=True)

    data_iter = filter(lambda x: x[0] < IOU_THRESHOLD, scan_iou.from_dataset(
        dataset.objdet.from_path(image_df()), sizes=SCALES,
        steps=list(map(lambda x: int(0.25 * x), SCALES))))

    data_iter = map(lambda x: x[1], data_iter)
    data_to_dump, data_iter = tee(data_iter, 2)
    data_iter = map(lambda x: from_array(x), data_iter)
    batches = batch(zip(data_iter, data_to_dump), batch_size=BATCH_SIZE)
    filt_data = map(lambda elmt: _filter_w_metadata(elmt, classifiers=classifiers), batches)
    imgs_to_dump = map(lambda x: x[1], chain.from_iterable(filt_data))
    imgs_to_path(
        negative_storage_path,
        islice(imgs_to_dump, 0, MAX_NEGATIVE_EXAMPLES)
    )


if __name__ == '__main__':

    _extract_negatives(RUN_ID)
