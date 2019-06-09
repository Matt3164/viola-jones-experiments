import os
import numpy as np

from common.config import PREVIOUS_RUN_ID, IM_SIZE
from train.classifier import load_trained
from train.dataset import negatives
from train.preprocessing import preprocess

if __name__ == '__main__':

    dry_run = False

    for run_id in np.arange(0, PREVIOUS_RUN_ID + 1):

        clf = load_trained(run_id)

        neg_files = negatives()

        neg_feat_array = np.empty((len(neg_files), IM_SIZE,IM_SIZE))

        for idx, impath in enumerate(neg_files):

            neg_feat_array[idx, ::] = preprocess(impath)

        preds = clf.predict(neg_feat_array)

        print(np.unique(preds, return_counts=True))

        if not dry_run:

            for p, path in zip(preds, neg_files):
                if p == 0:
                    print("Deleting {}".format(path))
                    os.remove(path)
