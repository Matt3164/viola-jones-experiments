from itertools import tee
from typing import Tuple

from numpy import array
from sklearn.cluster import KMeans

from common.iotools import image
from train.path_utils import image_df
from train.datasets.object_detection import from_path


def _to_rel_hw(elmt: Tuple[Tuple[int,int], Tuple[int,int, int,int]])->Tuple[float, float]:

    shape, flat_rect = elmt


    return (flat_rect[1]-flat_rect[0])/shape[0], (flat_rect[3]-flat_rect[2])/shape[1]

if __name__ == '__main__':
    obj_iter = from_path(image_df())

    bbs = map(lambda x: (image.from_path(x[0]).shape[:2] ,x[1]), obj_iter)

    wh_iter = map(_to_rel_hw, bbs)

    forw, forh = tee(wh_iter, 2)

    h_iter = map(lambda x: x[0], forh)

    w_iter = map(lambda x: x[1], forw)

    H = array(list(h_iter))
    W = array(list(w_iter))

    km = KMeans(n_clusters=10)

    km.fit(H.reshape(-1,1))
    km.cluster_centers_.sort(axis=0)
    print((km.cluster_centers_*256))

    km = KMeans(n_clusters=10)

    km.fit(W.reshape(-1, 1))
    km.cluster_centers_.sort(axis=0)

    print((km.cluster_centers_ * 256))