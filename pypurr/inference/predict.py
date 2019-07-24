from itertools import chain, tee
from typing import List, Tuple, Iterator

import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.pipeline import Pipeline

import pypurr.common.helpers.model as helpers_model
from pypurr.common import preprocessing
from pypurr.common.config import SCALES, RUN_ID, THRESHOLD
from pypurr.common.helpers.functionnal.utils import _to_tuple
from pypurr.common.scanner import grid, scan
from pypurr.inference.model import ensemble
from pypurr.train.helpers.dataset.objdet import Point, Size2D
from pypurr.train.path_utils import classifier


def _load_model() -> Pipeline:
    classifiers = [helpers_model.from_path(classifier(run_id)) for run_id in np.arange(0, RUN_ID)]
    model = ensemble.from_clfs(classifiers, threshold=THRESHOLD)
    return model


def from_img(img: ndarray, model: Pipeline = None) -> List[Tuple[Point, Size2D]]:
    # bb_iter = list(
    #     chain.from_iterable([
    #         grid.from_image(img, _to_tuple(s), _to_tuple(int(0.25 * s))) for s in SCALES
    #     ]
    #     )
    # )

    scan_iter = scan.from_array(preprocessing.image.from_array(img), map(lambda s: _to_tuple(int(0.25*s)), SCALES), map(_to_tuple, SCALES))

    for_bbox_iter, for_arr_iter = tee(scan_iter)


    arr = np.array(list(
        map(preprocessing.window.from_array, map(lambda x: x[1], for_arr_iter))
    ))

    bb_iter = map(lambda x: x[0], for_bbox_iter)

    print(arr.shape)

    if not model:
        model = _load_model()

    preds = model.predict(arr)

    return list(
        map(
            lambda x: x[1], filter(lambda x: x[0], zip(preds, bb_iter))
        )
    )


def from_images(images: Iterator[ndarray], model: Pipeline = None) -> Iterator[List[Tuple[Point, Size2D]]]:
    if not model:
        model = _load_model()

    return map(lambda img: from_img(img, model), images)
