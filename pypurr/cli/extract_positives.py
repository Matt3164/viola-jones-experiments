from itertools import islice
from os import makedirs

from pypurr.common.config import POSITIVE_PATH, SCALES, IOU_THRESHOLD, MAX_POSITIVE_EXAMPLES
from pypurr.train import scan_iou
from pypurr.train.helpers import dataset
from pypurr.train.path_utils import image_df


def _extract_positives():
    makedirs(POSITIVE_PATH, exist_ok=True)

    data_iter = filter(lambda x: x[0] >= IOU_THRESHOLD, scan_iou.from_dataset(
                            dataset.objdet.from_path(image_df()),
                            sizes=SCALES,
                            steps=list(map(lambda x: int(0.25*x), SCALES))
                        )
                       )

    dataset.images.to_path(
        POSITIVE_PATH,
        islice(map(lambda x: x[1], data_iter), 0, MAX_POSITIVE_EXAMPLES)
    )

    # data_iter = dataset.objdet.from_path(image_df())

    # data_iter = map(lambda x: (image.from_path(x[0]), x[1]), data_iter)
    #
    # data_iter = map(lambda e: filter(lambda x: x[0]>=0.2,scan_iou.from_array(e[0], e[1], steps=list(map(_to_tuple, map(lambda x: int(0.25 * x), SCALES))),sizes=list(map(_to_tuple, SCALES)))
    #     )
    #                 , data_iter)
    #
    #
    # dataset.images.to_path(POSITIVE_PATH, map(lambda x: x[1], chain.from_iterable(data_iter)))

    # data_iter = scan_iou.from_dataset(
    #     data_iter,
    #     sizes=SCALES,
    #     steps=list(map(lambda x: int(0.25 * x), SCALES))
    # )
    #
    # data_iter = filter(lambda x: x[0]>=0.2, data_iter)
    #
    # dataset.images.to_path(POSITIVE_PATH, map(lambda x: x[1],data_iter))


if __name__ == '__main__':
    _extract_positives()
