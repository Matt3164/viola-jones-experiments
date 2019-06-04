from os.path import join
import cv2
from matplotlib.pyplot import subplot, imshow, show
from pandas import read_csv

from config import WORKSPACE
from utils.rectangle import _bbox_to_mask
from utils.viz import _overlay_bbox_on_img

if __name__ == '__main__':
    workspace = WORKSPACE

    dataframe_path = join(workspace, "images.csv")

    df = read_csv(dataframe_path, index_col=0)

    array_flow = map(
        lambda arr: cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
        map(lambda fn: cv2.imread(fn, cv2.IMREAD_COLOR), df.ImagePath)
    )

    bbox_flow = zip(df.Xmin, df.Xmax, df.Ymin, df.Ymax)

    for im, bbox in zip(array_flow, bbox_flow):

        mask = _overlay_bbox_on_img((im.copy(), bbox))

        subplot(1, 2, 1)
        imshow(im)
        subplot(1, 2, 2)
        imshow(mask)
        show()
