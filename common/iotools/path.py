from glob import glob
from os.path import join
from typing import List


def find_pngs(path: str)->List[str]:
    return glob(join(path, "*.png"))