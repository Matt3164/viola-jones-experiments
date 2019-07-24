from itertools import islice
from os import makedirs

from pypurr.common.config import POSITIVE_PATH, MAX_POSITIVE_EXAMPLES, IOU_THRESHOLD, SCALES
from pypurr.train import scan_iou
from pypurr.train.helpers import dataset
from pypurr.train.path_utils import image_df


def _extract_positives():

    makedirs(POSITIVE_PATH, exist_ok=True)
    data_iter = filter(lambda x: x[0] >= IOU_THRESHOLD, scan_iou.from_dataset(
                            dataset.objdet.from_path(image_df()),
                            sizes=SCALES,
                            steps=map(lambda x: int(0.25*x), SCALES)
                        )
                       )
    dataset.images.to_path(
        POSITIVE_PATH,
        islice(map(lambda x: x[1], data_iter), 0, MAX_POSITIVE_EXAMPLES)
    )


if __name__ == '__main__':
    _extract_positives()