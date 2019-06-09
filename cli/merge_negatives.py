import os
import shutil
from glob import glob
from os.path import join, exists

from common.config import NEGATIVE_PATH, RUN_ID, PREVIOUS_RUN_ID

if __name__ == '__main__':

    current_negative_paths = join(NEGATIVE_PATH, "{:04d}".format(RUN_ID))

    assert exists(current_negative_paths)

    if RUN_ID > 0:
        previous_run = PREVIOUS_RUN_ID
        previous_negative_paths = join(NEGATIVE_PATH, "{:04d}".format(previous_run))
        assert exists(previous_negative_paths)

        for absfn in glob(join(previous_negative_paths, "*.png")):

            fn = os.path.split(absfn)[-1]

            shutil.copyfile(
                join(previous_negative_paths, fn),
                join(current_negative_paths, fn),
            )
