from itertools import islice
from os import makedirs
from common.config import MAX_NEGATIVE_EXAMPLES, RUN_ID, IOU_THRESHOLD
from common.iotools import images
from train.extract import _cat_scan_iou
from train.path_utils import negatives


def _extract_raw_negatives(run_id: int):
    negative_storage_path = negatives(run_id)
    makedirs(negative_storage_path, exist_ok=True)
    data_iter = filter(lambda x: x[0] < IOU_THRESHOLD, _cat_scan_iou())
    images.to_path(negative_storage_path,
                   islice(map(lambda x: x[1], data_iter), MAX_NEGATIVE_EXAMPLES)
                   )


if __name__ == '__main__':

    _extract_raw_negatives(RUN_ID)