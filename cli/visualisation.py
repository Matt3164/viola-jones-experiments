from matplotlib.pyplot import subplot, imshow, show

from common.imatools import read_ima
from common.viz import overlay_bbox_on_img
from train.dataset import load

if __name__ == '__main__':
    dataset_iter = load()

    display_iter = map(
        lambda x: (x[0], overlay_bbox_on_img((x[0].copy(), x[1]))),
        map(lambda x: (read_ima(x[0]), x[1]), dataset_iter)
    )

    for img, mask in display_iter:

        subplot(1, 2, 1)
        imshow(img)
        subplot(1, 2, 2)
        imshow(mask)
        show()
