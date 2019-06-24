WORKSPACE = "/home/matthieu/Workspace/data/cats"
POSITIVE_PATH = "/home/matthieu/Workspace/data/cats/detector/positives"

NOMINAL_SIZE=32

N_SCALES=10
SCALE_EXPANSION_FACTOR=0.1

SCALES = [int(NOMINAL_SIZE*(1.+SCALE_EXPANSION_FACTOR)**scale) for scale in range(N_SCALES)]

MAX_POSITIVE_EXAMPLES = 1000
NEGATIVE_PATH = "/home/matthieu/Workspace/data/cats/detector/negatives"
MAX_NEGATIVE_EXAMPLES = 5000
RUN_PATH = "/home/matthieu/Workspace/data/cats/detector/runs"
RUN_ID = 4

RANDOM_SEARCH=True

IOU_THRESHOLD = 0.5
BATCH_SIZE = 1024
N_FEATURES_MAX=1500

PROCESS_SIZE = 300