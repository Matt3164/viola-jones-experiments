from typing import Iterator, Tuple

import attr
from numpy import concatenate, expand_dims
from numpy.core.multiarray import ndarray

from common.config import SCALES
from common.imatools import read
from common.scanner import scan_pyramid_image
from train.datasets.object import load
from train.rectangle import rect_to_mask

@attr.s
class ChipDetails(object):
    """"""

    metadata = attr.ib(type=dict, kw_only=True)
    iou = attr.ib(type=float, kw_only=True)

def scan_iou()->Iterator[Tuple[ndarray, ChipDetails]]:

    """


    Args:
        iou_fn:

    Returns:

        Iterator[scale, img_idx, bb_idx, extracted array]

    """

    dataset_iter = load()

    dataset_iter = map(lambda elmt: (read(elmt[0]), elmt[1]), dataset_iter)

    dataset_iter = map(lambda elmt: (elmt[0], rect_to_mask(rect=elmt[1], shape=elmt[0].shape[:2],
                                                           burn_values=1)), dataset_iter)

    dataset_iter = map(lambda elmt: to_composite(elmt[0], elmt[1]), dataset_iter)

    for idx, composite_image in enumerate(dataset_iter):

        bbox_area = composite_image[:, :, -1].sum()

        for bb_idx, (bb, extim) in enumerate(scan_pyramid_image(
                composite_image,
                sizes=SCALES,
                steps=list(map(lambda sc: int(0.25 * sc), SCALES)))):

            intersection = extim[:, :, -1].sum()

            scale = extim.shape[0]

            ext_bb_area = scale ** 2

            iou = intersection / (bbox_area + ext_bb_area - intersection)

            yield extim[:,:,:3],ChipDetails(
                iou=iou,
                metadata=dict(img_idx=idx, bb_idx=bb_idx, scale=scale)
            )


def to_composite(img: ndarray, mask: ndarray)->ndarray:
    return concatenate([img, expand_dims(mask, axis=-1)], axis=-1)