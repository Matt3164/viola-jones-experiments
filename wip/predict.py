from itertools import tee
from os.path import join
from typing import List, Tuple

import cv2
from matplotlib.pyplot import subplot, imshow, show
from pandas import read_csv, DataFrame

from common.config import WORKSPACE, RUN_PATH, SCALES, RUN_ID
from common.filter import by_cascade
from common.iotools.image import from_path
from common.scanner import scan_image
from common.viz import overlay_bbox_on_img
import numpy as np

from inference.preprocessing import from_array
from train.preprocessing.raw import _downscale
from train.models.utils import from_path
from train.datasets.object import from_path as iter_from_path
from train.path_utils import classifier, image_df


def bbs(image_arr):
    for scale in SCALES:
        for bb, ext_arr in scan_image(
                image_arr,
                step=int(0.25 * scale),
                size=scale
        ):
            yield bb


def preprocess(image):
    image = cv2.resize(image, (32, 32))
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def _extract_bb(arr: np.ndarray, bb: Tuple[int,int,int,int])->np.ndarray:
    return arr[bb[0]:(bb[0] + bb[2]), bb[1]:(bb[1] + bb[3]), :]

def _predict(img_arr: np.ndarray)->Tuple[np.ndarray, Tuple[int,int,int,int]]:

    boundingboxes = list(bbs(img_arr))
    arr = np.array(list(
        map(preprocess, map(lambda bb: _extract_bb(img_arr, bb), boundingboxes))
    ))

    print(arr.shape)

    classifiers = [from_path(classifier(run_id)) for run_id in np.arange(0, RUN_ID)]

    kept_arr, metadata, _, _ = by_cascade(arr, boundingboxes, classifiers)

    return kept_arr, metadata

if __name__ == '__main__':

    data_iter = iter_from_path(image_df(), sample=True, n_samples=1)

    data_iter = map(lambda x: (from_array(from_path(x[0])), x[1]), data_iter)

    arr, meta = next(data_iter)

    kept_arr, metadata = _predict(arr)

    pmask = overlay_bbox_on_img((arr.copy(), meta))

    for bb in metadata:
        pmask = overlay_bbox_on_img((pmask, (bb[1], bb[1] + bb[3], bb[0], bb[0] + bb[2])), color=(0, 255, 0))

    subplot(1, 2, 1)
    imshow(arr)
    subplot(1, 2, 2)
    imshow(pmask)
    show()




