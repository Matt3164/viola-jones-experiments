from os import makedirs

from common.config import RUN_ID, POSITIVE_PATH
from common.iotools.dataset import to_path, from_paths
from common.iotools.labeled_path import from_folders
from train.path_utils import run, dataset, negatives

def _build_dataset(run_id: int)->None:
    data = list(from_folders([negatives(run_id), POSITIVE_PATH]))

    X, Y = from_paths(data)

    makedirs(run(run_id), exist_ok=True)

    to_path(dataset(run_id), X, Y)


if __name__ == '__main__':

    _build_dataset(RUN_ID)