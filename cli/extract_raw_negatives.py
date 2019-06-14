from os import makedirs
from os.path import join, exists

import cv2

from common.config import MAX_NEGATIVE_EXAMPLES, RUN_ID, IOU_THRESHOLD
from common.imatools import write
from train.extract import scan_iou
from train.path_utils import negatives

if __name__ == '__main__':

    negative_storage_path = negatives(RUN_ID)

    makedirs(negative_storage_path, exist_ok=True)

    data_iter = filter(lambda x: x[1].iou < IOU_THRESHOLD, scan_iou())

    for idx, (arr, details) in enumerate(data_iter):

        pos_fn = join(negative_storage_path, "ext_{0}_scale_{1}_{2:05d}.png".format(
            details.metadata["img_idx"],
            details.metadata["scale"],
            details.metadata["bb_idx"]))

        if not exists(pos_fn):
            write(pos_fn, arr)

        if idx > MAX_NEGATIVE_EXAMPLES:
            break