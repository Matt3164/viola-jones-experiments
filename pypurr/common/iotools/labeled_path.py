from itertools import chain, repeat
from typing import List, Iterator, Tuple

from pypurr.common.iotools.path import find_pngs


def from_folders(folders: List[str])->Iterator[Tuple[str, int]]:
    """


    Returns:
        Data iterator: path, label

    """
    return chain.from_iterable(
        map(lambda elmt: zip(elmt[1], repeat(elmt[0])), enumerate(map(find_pngs, folders)))
    )