from itertools import tee, chain
from typing import Iterator, Tuple

import attr
from numpy import concatenate, expand_dims
from numpy.core._multiarray_umath import ndarray
from numpy.core.multiarray import ndarray

from common.config import SCALES
from common.iotools.image import from_path
from common.imgproc.scan import from_image_pyramid
from train.datasets.object_detection import _load
from train.preprocessing.raw import from_array
from train.rectangle import rect_to_mask

def _cat_scan_iou()->Iterator[Tuple[float, ndarray]]:

    """


    Args:
        iou_fn:

    Returns:

        Iterator[scale, img_idx, bb_idx, extracted array]

    """

    return scan_iou_from_dataset(_load())


def scan_iou_from_dataset(dataset_iter: Iterator[Tuple[str, Tuple[int,int,int,int]]])->Iterator[Tuple[float, ndarray]]:

    """


    Args:
        iou_fn:

    Returns:

        Iterator[scale, img_idx, bb_idx, extracted array]

    """

    dataset_iter = map(lambda elmt: (from_path(elmt[0]), elmt[1]), dataset_iter)

    dataset_iter = map(lambda elmt: (elmt[0], rect_to_mask(rect=elmt[1], shape=elmt[0].shape[:2],
                                                           burn_values=1)), dataset_iter)

    dataset_iter = map(lambda elmt: to_composite(elmt[0], elmt[1]), dataset_iter)

    dataset_iter = map(preproc_from_composite, dataset_iter)

    return chain.from_iterable(
        map(_scan_iou_from_composite, dataset_iter)
    )


def _iou_from_composite(extim: ndarray, bbox_area: int)->float:
    intersection = extim[:, :, -1].sum()
    scale = extim.shape[0]
    ext_bb_area = scale ** 2
    iou = intersection / (bbox_area + ext_bb_area - intersection)
    return iou

def _scan_iou_from_composite(composite_image: ndarray)->Iterator[Tuple[float, ndarray]]:
    bbox_area = composite_image[:, :, -1].sum()

    ext_im_iter = from_image_pyramid(
        composite_image,
        sizes=SCALES,
        steps=list(map(lambda sc: int(0.125 * sc), SCALES)))

    for_iou, data_iter = tee(map(lambda x: x[1], ext_im_iter), 2)

    iou_iter = map(lambda x: _iou_from_composite(x, bbox_area), for_iou)

    for iou, arr in zip(iou_iter, map(lambda x: x[:, :, :3], data_iter)):
        yield iou, arr

def to_composite(img: ndarray, mask: ndarray)->ndarray:
    return concatenate([img, expand_dims(mask, axis=-1)], axis=-1)


def preproc_from_composite(composite: ndarray)->ndarray:
    return from_array(composite)