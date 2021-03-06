from os import makedirs

from pypurr.common.config import RUN_ID, POSITIVE_PATH
from pypurr.deprecated.iotools.dataset import from_paths, to_path
from pypurr.deprecated.iotools.labeled_path import from_folders
from pypurr.train.path_utils import run, dataset, negatives


def _build_dataset(run_id: int) -> None:
    data = list(from_folders([negatives(run_id), POSITIVE_PATH]))

    X, Y = from_paths(data)

    makedirs(run(run_id), exist_ok=True)

    to_path(dataset(run_id), X, Y)


if __name__ == '__main__':

    _build_dataset(RUN_ID)
