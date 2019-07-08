import cv2
from cv2 import resize
from numpy import concatenate
from skimage.data import chelsea

"""

Try every haar feature and keep the bests.

"""
if __name__ == '__main__':
    im = chelsea()

    im = concatenate([im[:,:,:2], im[:,:,1:]], axis=-1)

    im = resize(im, (256, 256))

    print(im.shape)

    cv2.imshow("Cat", im[:,:,:3])
    cv2.waitKey(0)

