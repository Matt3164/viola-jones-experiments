import os
import shutil
from glob import glob
from os.path import join

import cv2
import numpy as np
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from sklearn.externals import joblib

from train import *
from config import NEGATIVE_PATH, RUN_PATH, RUN_ID, PREVIOUS_RUN_ID

if __name__ == '__main__':

    dry_run = False

    for run_id in np.arange(0, PREVIOUS_RUN_ID + 1):

        clf_path = join(RUN_PATH, "{:04d}".format(run_id), "classifier.pickle")

        clf_dataset = np.load(join(RUN_PATH, "{:04d}".format(run_id), "dataset.npz"))

        neg_files = glob(join(NEGATIVE_PATH, "{:04d}".format(RUN_ID), "*.png"))

        neg_feat_array = np.empty((len(neg_files), 28,28))

        for idx, impath in enumerate(neg_files):
            image = cv2.imread(impath)

            image = cv2.resize(image, (28, 28))

            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            image = integral_image(image)

            neg_feat_array[idx, ::] = image

        clf = joblib.load(clf_path)

        preds = clf.predict(neg_feat_array)

        print(np.unique(preds, return_counts=True))

        if not dry_run:

            for p, path in zip(preds, neg_files):
                if p == 0:
                    print("Deleting {}".format(path))
                    os.remove(path)
