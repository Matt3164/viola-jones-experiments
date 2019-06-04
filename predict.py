from os.path import join

import cv2
from matplotlib.pyplot import subplot, imshow, show
from pandas import read_csv, DataFrame

from config import WORKSPACE
from utils.rectangle import _bbox_to_mask
from utils.scanner import scan_image
from utils.viz import _overlay_bbox_on_img

if __name__ == '__main__':
    dataframe_path = join(WORKSPACE, "images.csv")

    df = read_csv(dataframe_path, index_col=0)

    df = df.sample(n=1)

    assert isinstance(df, DataFrame)

    elmt = df._ixs(0, axis=0)

    image_path = elmt.ImagePath

    image_arr = cv2.cvtColor(
        cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

    mask = _overlay_bbox_on_img((image_arr.copy(), (elmt.Xmin, elmt.Xmax, elmt.Ymin, elmt.Ymax)))

    scale = 128

    for bb, ext_arr in scan_image(
            image_arr,
            step=int(0.25 * scale),
            size=scale
    ):
        mask = _overlay_bbox_on_img((mask, (bb[1], bb[1] + bb[3], bb[0], bb[0] + bb[2])), color=(0, 255, 0))

    subplot(1, 2, 1)
    imshow(image_arr)
    subplot(1, 2, 2)
    imshow(mask)
    show()
