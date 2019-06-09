from typing import Iterator, Callable, Tuple

from numpy import concatenate, expand_dims
from numpy.core._multiarray_umath import ndarray

from numpy.core.multiarray import ndarray

from common.config import SCALES
from common.imatools import read_ima
from common.scanner import scan_image_multiple_scales
from train.dataset import load
from train.rectangle import rect_to_mask


def scan_iou(iou_fn: Callable[[float], bool])->Iterator[Tuple[int, int,int , ndarray]]:

    """


    Args:
        iou_fn:

    Returns:

        Iterator[scale, img_idx, bb_idx, extracted array]

    """

    dataset_iter = load()

    dataset_iter = map(lambda elmt: (read_ima(elmt[0]), elmt[1]), dataset_iter)

    dataset_iter = map(lambda elmt: (elmt[0], rect_to_mask(rect=elmt[1], shape=elmt[0].shape[:2],
                                                           burn_values=1)), dataset_iter)

    dataset_iter = map(lambda elmt: to_composite(elmt[0], elmt[1]), dataset_iter)

    for idx, composite_image in enumerate(dataset_iter):

        bbox_area = composite_image[:, :, -1].sum()

        for bb_idx, (bb, extim) in enumerate(scan_image_multiple_scales(
                composite_image,
                sizes=SCALES,
                steps=list(map(lambda sc: int(0.25 * sc), SCALES)))):

            intersection = extim[:, :, -1].sum()

            scale = extim.shape[0]

            ext_bb_area = scale ** 2

            iou = intersection / (bbox_area + ext_bb_area - intersection)

            if iou_fn(iou):
                yield scale, idx, bb_idx, extim[:,:,:3]


def to_composite(img: ndarray, mask: ndarray)->ndarray:
    return concatenate([img, expand_dims(mask, axis=-1)], axis=-1)