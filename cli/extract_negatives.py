from os.path import join, exists

import cv2

from common.config import NEGATIVE_PATH, MAX_NEGATIVE_EXAMPLES, RUN_ID, IOU_THRESHOLD
from train.extract import scan_iou

if __name__ == '__main__':

    negative_storage_path = join(NEGATIVE_PATH, "{:04d}".format(RUN_ID))

    for idx, (scale, img_idx, bb_idx, ext_arr) in enumerate(scan_iou(lambda x: x < IOU_THRESHOLD)):

        pos_fn = join(negative_storage_path , "ext_{0}_scale_{1}_{1:5d}.png".format(img_idx, scale, bb_idx))

        if not exists(pos_fn):
            cv2.imwrite(
                pos_fn,
                cv2.cvtColor(ext_arr, cv2.COLOR_RGB2BGR)
            )

        if idx > MAX_NEGATIVE_EXAMPLES:
            break