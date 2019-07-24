from typing import Iterator, Tuple
from pandas import DataFrame, read_csv

Point = Tuple[int, int]
Size2D = Tuple[int, int]
Rect=Tuple[Point, Size2D]

def from_df(df: DataFrame, shuffle: bool=True, sample: bool=False, n_samples: int=0)->Iterator[Tuple[str, Rect]]:

    if shuffle:
        df = df.sample(frac=1.)

    if sample:
        df = df.sample(n=n_samples)

    assert isinstance(df, DataFrame)

    for idx, elmt in df.iterrows():
        yield elmt.ImagePath, ( (elmt.Ymin, elmt.Xmin), (elmt.Ymax - elmt.Ymin, elmt.Xmax - elmt.Xmin))


def from_path(path: str)->Iterator[Tuple[str, Rect]]:
    return from_df(read_csv(path, index_col=0))