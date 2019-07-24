from itertools import tee
from typing import Tuple

from numpy import array, unique
from sklearn.cluster import KMeans

from pypurr.common.helpers import image
from pypurr.train.helpers import dataset
from pypurr.train.helpers.dataset.objdet import Rect
from pypurr.train.path_utils import image_df



def _to_rel_hw(elmt: Tuple[Tuple[int,int], Rect])->Tuple[float, float]:
    shape, rect = elmt
    return rect[1][1]/shape[0], rect[1][0]/shape[1]


def _compute_scales():
    obj_iter = dataset.objdet.from_path(image_df())

    bbs = map(lambda x: (image.from_path(x[0]).shape[:2], x[1]), obj_iter)

    wh_iter = map(_to_rel_hw, bbs)
    forw, forh = tee(wh_iter, 2)
    h_iter = map(lambda x: x[0], forh)
    w_iter = map(lambda x: x[1], forw)
    H = array(list(h_iter))
    W = array(list(w_iter))
    km = KMeans(n_clusters=10)
    print("H Scale : ")
    km.fit(H.reshape(-1, 1))
    km.cluster_centers_.sort(axis=0)
    print((km.cluster_centers_ * 256))
    print(unique(km.predict(H.reshape(-1, 1)), return_counts=True))
    km = KMeans(n_clusters=10)
    km.fit(W.reshape(-1, 1))
    km.cluster_centers_.sort(axis=0)
    print("V Scale : ")
    print((km.cluster_centers_ * 256))
    print(unique(km.predict(W.reshape(-1, 1)), return_counts=True))


if __name__ == '__main__':
    _compute_scales()