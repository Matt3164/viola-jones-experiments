from itertools import chain, repeat
from typing import Tuple, Iterator, List

from common.imatools import glob_png


def from_folders(folders: List[str])->Iterator[Tuple[str, int]]:
    """


    Returns:
        Data iterator: path, label

    """
    return chain.from_iterable(
        map( lambda elmt: zip(elmt[1], repeat(elmt[0])) , enumerate(map(glob_png, folders)) )
    )

