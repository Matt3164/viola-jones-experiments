from itertools import chain
from os import makedirs
from os.path import exists, join
from typing import List, Tuple, Iterator
from numpy.core.multiarray import ndarray
from sklearn.base import BaseEstimator
from common.config import MAX_NEGATIVE_EXAMPLES, IOU_THRESHOLD, BATCH_SIZE
from common.filter import by_cascade
from common.imatools import write
from common.iter import batch
from train.classifier import from_path as clf_from_path
from train.extract import scan_iou, ChipDetails
from train.path_utils import negatives, classifier
from cli.train_cli import *
from train.preprocessing import from_array
import numpy as np


def _filter_w_metadata(data: List[Tuple[ndarray, ChipDetails]],
                       classifiers: List[BaseEstimator]
                       )->Iterator[Tuple[ndarray, ChipDetails]]:

    X = np.array(list(map(lambda x: x[0], data)))
    keptX, keptMeta, _, _ = by_cascade(X, list(map(lambda x: x[1], data)), classifiers)
    return zip(keptX, keptMeta)

def _add_image_to_metadata(elmt: Tuple[ndarray, ChipDetails])->Tuple[ndarray, ChipDetails]:
    elmt[1].metadata["origin_array"]=elmt[0]
    return elmt[0],elmt[1]

if __name__ == '__main__':

    negative_storage_path = negatives(RUN_ID)

    classifiers = [clf_from_path(classifier(run_id)) for run_id in np.arange(0, RUN_ID)]

    makedirs(negative_storage_path, exist_ok=True)

    data_iter = filter(lambda x: x[1].iou < IOU_THRESHOLD, scan_iou())

    data_iter = map(_add_image_to_metadata, data_iter)

    data_iter = map(lambda x: (from_array(x[0]), x[1]), data_iter)

    batches = batch(data_iter, batch_size=BATCH_SIZE)

    filt_data = map(lambda elmt: _filter_w_metadata(elmt, classifiers=classifiers), batches)

    for idx, (data, details) in enumerate(chain.from_iterable(filt_data)):

        pos_fn = join(negative_storage_path, "ext_{0}_scale_{1}_{2:05d}.png".format(
            details.metadata["img_idx"],
            details.metadata["scale"],
            details.metadata["bb_idx"]))

        if not exists(pos_fn):
            write(pos_fn, details.metadata["origin_array"])

        if idx > MAX_NEGATIVE_EXAMPLES:
            break