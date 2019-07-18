from itertools import tee, islice
from os import makedirs
from os.path import join
from typing import Tuple, List

import cv2
import numpy as np
from matplotlib.pyplot import show, imshow, title
from numpy.core.multiarray import ndarray

from pypurr.common.config import RUN_ID, SCALES
from pypurr.common.func.utils import _to_tuple
from pypurr.common.imgproc import pyramid
from pypurr.common.iotools import image
from pypurr.common.iotools.image import from_path, crop
from pypurr.common.viz import overlay_bbox_on_image
from pypurr.inference.model import aggregation
from pypurr.inference.predict import Point, Size2D
from pypurr.inference.preprocessing import from_array, from_window
from pypurr.train.datasets.object_detection import from_path as iter_from_path
from pypurr.train.models import utils
from pypurr.train.path_utils import image_df, classifier, run

N_IMAGES = 5


def _display_bbs(img: ndarray, bbs: List[Tuple[Point, Size2D]]) -> ndarray:
    _img = img.copy()
    for bb in bbs:
        _img = overlay_bbox_on_image(_img, bb, color=(0, 255, 0))
    return _img


def _to_rect(flat: Tuple[int, int, int, int]) -> Tuple[Point, Size2D]:
    return (flat[2], flat[0]), (flat[3] - flat[2], flat[1] - flat[0])


if __name__ == '__main__':

    data_iter = iter_from_path(image_df(), shuffle=True)

    data_iter = map(lambda x: (from_path(x[0]), x[1]), data_iter)

    gt_data, to_pred_data = tee(data_iter, 2)

    true_data = map(lambda x: (x[0], _to_rect(x[1])), gt_data)
    true_data = map(lambda x: overlay_bbox_on_image(x[0], x[1], color=(255, 0, 0)), true_data)

    classifiers = [utils.from_path(classifier(run_id)) for run_id in np.arange(0, RUN_ID)]

    # for (im, path) in to_pred_data:

    dump_folder = run(RUN_ID)

    dump_path = join(dump_folder, "predictions")
    makedirs(dump_path, exist_ok=True)

    for idx, (im, path) in enumerate(islice(to_pred_data, 10)):
        img = from_array(im)
        print(img.shape)

        boundingboxes = list(
            pyramid.from_image(img, map(_to_tuple, SCALES), map(_to_tuple, map(lambda x: int(0.125 * x), SCALES))))

        arr = np.array(list(
            map(from_window, map(lambda bb: crop(img, bb[0], bb[1]), boundingboxes))
        ))

        print(arr.shape)


        for thd in [0.5, 0.75, 0.9, 0.95, 0.99]:

            print(thd)

            pipe = aggregation.from_clfs(classifiers, threshold=thd)

            pred = pipe.transform(arr)

            bbs = list(
                map(
                    lambda x: x[1],
                    filter(
                        lambda x: x[0],
                        zip(pred, boundingboxes))

                ))

            print("{0}/{1}".format(len(bbs), len(boundingboxes)))

            image.to_path(join(dump_folder, "ensemble_{0}_{1}.png".format(thd, idx)), _display_bbs(img, bbs))

        # for thd in [0.5, 0.75, 0.9, 0.95, 0.99]:
        #     bbs = list(
        #         map(
        #             lambda x: x[1],
        #             filter(
        #                 lambda x: x[0],
        #                 zip((pred[:,::2]>0.5).mean(axis=1)>thd, boundingboxes))
        #
        #         ))
        #
        #     print("{0}/{1}".format(len(bbs), len(boundingboxes)))

            # imshow(_display_bbs(img, bbs))
            # title("Ensemble disc. {}".format(thd))
            # show()

            # image.to_path(join(dump_folder, "ensemble_disc_{0}_{1}.png".format(thd, idx)), _display_bbs(img, bbs))

        # pred_data = map(lambda x: (x[0], from_img(from_array(x[0]))), to_pred_data)
        #
    # pred_data = map(lambda x: _display_bbs(x[0], x[1]), pred_data)
    #
    # dump_folder = run(RUN_ID)
    #
    # for tag, images in zip(["gt", "prediction"], [true_data, pred_data]):
    #     images = islice(images, 0, N_IMAGES)
    #
    #     dump_path = join(dump_folder, tag)
    #     makedirs(dump_path, exist_ok=True)
    #     to_path(dump_path, images)
