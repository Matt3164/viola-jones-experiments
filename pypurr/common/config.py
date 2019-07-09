import os
from os.path import join

from dotenv import load_dotenv

load_dotenv(verbose=True, override=False)

WORKSPACE = os.environ.get('WORKSPACE')

DATA_PATH = os.environ.get('DATA_PATH')

EXP_NAME=os.environ.get('EXP_NAME')

POSITIVE_PATH = join(
    WORKSPACE, EXP_NAME, "positives")

NEGATIVE_PATH = join(WORKSPACE, EXP_NAME, "negatives")

RUN_PATH = join(WORKSPACE, EXP_NAME, "runs")

NOMINAL_SIZE=32

N_SCALES=10
SCALE_EXPANSION_FACTOR=0.1

SCALES = [int(NOMINAL_SIZE*(1.+SCALE_EXPANSION_FACTOR)**scale) for scale in range(N_SCALES)]

MAX_POSITIVE_EXAMPLES = 1000

MAX_NEGATIVE_EXAMPLES = 5000

RUN_ID = 5

RANDOM_SEARCH=True

IOU_THRESHOLD = 0.5
BATCH_SIZE = 1024
N_FEATURES_MAX=1500

PROCESS_SIZE = 300