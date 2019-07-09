from os.path import join

from pypurr.common.config import RUN_PATH, NEGATIVE_PATH, WORKSPACE

DATASET_FILE = "dataset.npz"

IMAGE_DF = "images.csv"

CLASSIFIER_FILE = "classifier.pickle"


def classifier(run_id: int)->str:
    return join(RUN_PATH, run_tag(run_id), CLASSIFIER_FILE)


def run_tag(run_id):
    return "{:04d}".format(run_id)


def negatives(run_id: int)->str:
    return join(NEGATIVE_PATH, run_tag(run_id))


def run(run_id)->str:
    return join(RUN_PATH, run_tag(run_id))


def image_df()->str:
    return join(WORKSPACE, IMAGE_DF)


def dataset(run_id: int)->str:
    return join(run(run_id), DATASET_FILE)