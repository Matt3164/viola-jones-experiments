from typing import List, Tuple, Iterator

from numpy.core.multiarray import ndarray
import numpy as np
from sklearn.pipeline import Pipeline

from pypurr.common.config import SCALES, RUN_ID, THRESHOLD
from pypurr.common.filter import by_clf
from pypurr.common.func.utils import _to_tuple
from pypurr.common.imgproc import pyramid
from pypurr.common.iotools.image import crop
from pypurr.inference.model import aggregation
from pypurr.inference.preprocessing import from_window
from pypurr.train.models import utils
from pypurr.train.path_utils import classifier


def _load_model()->Pipeline:
    classifiers = [utils.from_path(classifier(run_id)) for run_id in np.arange(0, RUN_ID)]
    model = aggregation.from_clfs(classifiers, threshold=THRESHOLD)
    return model


Point = Tuple[int, int]

Size2D = Tuple[int, int]


def from_img(img: ndarray, model: Pipeline=None)->List[Tuple[Point, Size2D]]:

    boundingboxes = list(
        pyramid.from_image(img, map(_to_tuple, SCALES), map(_to_tuple, map(lambda x: int(0.25 * x), SCALES))))

    arr = np.array(list(
        map(from_window, map(lambda bb: crop(img, bb[0], bb[1]), boundingboxes))
    ))

    print(arr.shape)

    if not model:
        model = _load_model()

    kept_arr, metadata, _, _ = by_clf(arr, boundingboxes, model)

    return metadata


def from_images(images: Iterator[ndarray], model: Pipeline=None)->Iterator[List[Tuple[Point, Size2D]]]:

    if not model:
        model = _load_model()

    return map(lambda img: from_img(img, model), images)