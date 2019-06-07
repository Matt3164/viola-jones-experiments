from os.path import join

import cv2
import joblib
from matplotlib.pyplot import subplot, imshow, show
from pandas import read_csv, DataFrame

from config import WORKSPACE, PREVIOUS_RUN_ID, RUN_PATH
from utils.rectangle import rect_to_mask
from utils.scanner import scan_image
from utils.viz import overlay_bbox_on_img
import numpy as np

from train import *

def bbs(image_arr):
    for scale in [64, 128, 256, 512]:
        for bb, ext_arr in scan_image(
                image_arr,
                step=int(0.125 * scale),
                size=scale
        ):
            yield bb

def preprocess(image):
    image = cv2.resize(image, (28, 28))

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

if __name__ == '__main__':
    dataframe_path = join(WORKSPACE, "images.csv")

    df = read_csv(dataframe_path, index_col=0)

    df = df.sample(n=1)

    assert isinstance(df, DataFrame)

    elmt = df._ixs(0, axis=0)

    image_path = elmt.ImagePath

    image_arr = cv2.cvtColor(
        cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

    mask = overlay_bbox_on_img((image_arr.copy(), (elmt.Xmin, elmt.Xmax, elmt.Ymin, elmt.Ymax)))

    bounndingboxes = list(bbs(image_arr))

    for bb in bounndingboxes:
            mask = overlay_bbox_on_img((mask, (bb[1], bb[1] + bb[3], bb[0], bb[0] + bb[2])), color=(0, 255, 0))

    subplot(1, 2, 1)
    imshow(image_arr)
    subplot(1, 2, 2)
    imshow(mask)
    show()

    arr = np.array(list(
        map(preprocess, map(lambda bb: image_arr[bb[0]:bb[0]+bb[2], bb[1]:bb[1]+bb[3],:], bounndingboxes))
    ))

    print(arr.shape)

    for run_id in np.arange(0, PREVIOUS_RUN_ID + 1):

        clf_path = join(RUN_PATH, "{:04d}".format(run_id), "classifier.pickle")

        clf = joblib.load(clf_path)

        pred = clf.predict(arr)

        arr = arr[pred==1, ::]

        bounndingboxes = list(filter(lambda e: e[1]==1, zip(bounndingboxes, pred)))


    print(bounndingboxes)



