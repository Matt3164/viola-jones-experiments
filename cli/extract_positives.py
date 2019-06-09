from os.path import join, exists

import cv2

from common.config import POSITIVE_PATH, MAX_POSITIVE_EXAMPLES, IOU_THRESHOLD
from train.extract import scan_iou

if __name__ == '__main__':

    for idx, (scale, img_idx, bb_idx, ext_arr) in enumerate(scan_iou(lambda x: x>=IOU_THRESHOLD)):

        pos_fn = join(POSITIVE_PATH, "ext_{0}_scale_{1}_{1:5d}.png".format(img_idx, scale, bb_idx))

        if not exists(pos_fn):
            cv2.imwrite(
                pos_fn,
                cv2.cvtColor(ext_arr, cv2.COLOR_RGB2BGR)
            )

        if idx > MAX_POSITIVE_EXAMPLES:
            break