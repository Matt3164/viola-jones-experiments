from itertools import islice
from pypurr.common.config import POSITIVE_PATH, MAX_POSITIVE_EXAMPLES, IOU_THRESHOLD
from pypurr.common.iotools import images
from pypurr.train.extract import _cat_scan_iou


def _extract_positives():
    data_iter = filter(lambda x: x[1].iou >= IOU_THRESHOLD, _cat_scan_iou())
    images.to_path(
        POSITIVE_PATH,
        islice(map(lambda x: x[0], data_iter), 0, MAX_POSITIVE_EXAMPLES)
    )


if __name__ == '__main__':
    _extract_positives()