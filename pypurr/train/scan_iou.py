from itertools import chain, repeat
from typing import Iterator, List, Tuple

from imgaug import BoundingBox
from numpy.core.multiarray import ndarray

from pypurr.common.helpers import image
from pypurr.common.helpers.functionnal.utils import _to_tuple
from pypurr.common.scanner import scan
from pypurr.train.helpers.dataset.objdet import Rect, Size2D


def _rect_to_bbox(rect: Rect)->BoundingBox:
    return BoundingBox(
        x1=max(rect[0][1],0),
        y1=max(rect[0][0],0),
        x2=rect[0][1]+rect[1][1],
        y2=rect[0][0]+rect[1][0],
    )

def iou(rect1: Rect, rect2: Rect)->float:
    bb1 =_rect_to_bbox(rect1)
    bb2 = _rect_to_bbox(rect2)

    return bb1.iou(bb2)


def from_array(arr: ndarray, gtrect: Rect, steps: List[Size2D], sizes: List[Size2D])->Iterator[Tuple[float, ndarray]]:
    bb_gt_arr_iter = map(
        lambda x: (x[0][0], x[1], x[0][1]),
        zip(scan.from_array(arr, steps, sizes), repeat(gtrect))
    )

    return map(lambda elmt: (iou(elmt[0], elmt[1]), elmt[2]), bb_gt_arr_iter)

def from_dataset(dataset: Iterator[Tuple[str, Rect]], sizes: List[int], steps: List[int])->Iterator[Tuple[float, ndarray]]:

    dataset = map(lambda x: (image.from_path(x[0]), x[1]), dataset)

    return chain.from_iterable(
        map(lambda x: from_array(x[0], x[1], steps=list(map(_to_tuple, steps)), sizes=list(map(_to_tuple, sizes))), dataset)
    )