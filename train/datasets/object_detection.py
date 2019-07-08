from typing import Iterator, Tuple

from pandas import read_csv, DataFrame

from train.path_utils import image_df


def from_path(fn: str, shuffle: bool=False, sample: bool=False, n_samples: int=100)->Iterator[Tuple[str, Tuple[int,int,int,int]]]:

    df = read_csv(fn, index_col=0)

    if shuffle:
        df = df.sample(frac=1.)

    if sample:
        df = df.sample(n=n_samples)

    assert isinstance(df, DataFrame)

    for idx, elmt in df.iterrows():

        yield elmt.ImagePath, (elmt.Xmin, elmt.Xmax, elmt.Ymin, elmt.Ymax)


def _load(shuffle: bool=False, sample: bool=False, n_samples: int=100)->Iterator[Tuple[str, Tuple[int, int, int, int]]]:
    return from_path(image_df(), shuffle=False, sample=sample, n_samples=n_samples)