from cv2 import resize
from numpy.core.multiarray import ndarray
from skimage.data import chelsea
from skimage.feature import hog

from pypurr.train.models.hog.settings import CELL_PER_BLOCK, PIXELS_BY_CELL, ORIENTATIONS


def HOG(img: ndarray)->ndarray:
    """"""

    return hog(img,
               orientations=ORIENTATIONS,
               pixels_per_cell=(PIXELS_BY_CELL, PIXELS_BY_CELL),
               cells_per_block=(CELL_PER_BLOCK, CELL_PER_BLOCK),
               transform_sqrt=True,
               visualize=False,
               multichannel=False,
               feature_vector=True,
               block_norm="L2-Hys")


if __name__ == '__main__':
    im = resize(chelsea(), (24, 24))

    print(HOG(im).shape)




