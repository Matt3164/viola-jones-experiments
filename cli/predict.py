from itertools import tee, islice
from os import makedirs
from os.path import join
from typing import Tuple, List
from numpy.core.multiarray import ndarray

from common.config import RUN_ID
from common.iotools.image import from_path
from common.iotools.images import to_path
from common.viz import overlay_bbox_on_image
from inference.predict import from_img, Point, Size2D
from inference.preprocessing import from_array
from train.datasets.object_detection import from_path as iter_from_path
from train.path_utils import image_df, run

N_IMAGES = 5

def _display_bbs(img: ndarray, bbs: List[Tuple[Point, Size2D]])->ndarray:
    _img = img.copy()
    for bb in bbs:
        _img = overlay_bbox_on_image(_img, bb, color=(0, 255, 0))
    return _img

def _to_rect(flat: Tuple[int,int,int,int])->Tuple[Point, Size2D]:
    return (flat[2], flat[0]), (flat[3]-flat[2], flat[1]-flat[0])


if __name__ == '__main__':

    data_iter = iter_from_path(image_df(), shuffle=True)

    data_iter = map(lambda x: (from_path(x[0]), x[1]), data_iter)

    gt_data, to_pred_data = tee(data_iter, 2)

    true_data = map(lambda x: (x[0], _to_rect(x[1])), gt_data)
    true_data = map(lambda x: overlay_bbox_on_image(x[0], x[1], color=(255,0,0)), true_data)

    pred_data = map(lambda x: (x[0], from_img(from_array(x[0]))), to_pred_data)

    pred_data = map(lambda x: _display_bbs(x[0], x[1]), pred_data)

    dump_folder = run(RUN_ID)

    for tag, images in zip(["gt", "prediction"], [true_data, pred_data]):
        images = islice(images, 0, N_IMAGES)

        dump_path = join(dump_folder, tag)
        makedirs(dump_path, exist_ok=True)
        to_path(dump_path, images)
