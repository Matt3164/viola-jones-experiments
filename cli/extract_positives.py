from os.path import join, exists

import cv2

from common.config import POSITIVE_PATH, MAX_POSITIVE_EXAMPLES, IOU_THRESHOLD
from train.extract import scan_iou

if __name__ == '__main__':

    data_iter = filter(lambda x: x[1].iou >= IOU_THRESHOLD, scan_iou())

    for idx, (data, details) in enumerate(data_iter):

        pos_fn = join(POSITIVE_PATH, "ext_{0}_scale_{1}_{2:05d}.png".format(
            details.metadata["img_idx"],
            details.metadata["scale"],
            details.metadata["bb_idx"]))

        if not exists(pos_fn):
            cv2.imwrite(
                pos_fn,
                cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            )

        if idx > MAX_POSITIVE_EXAMPLES:
            break