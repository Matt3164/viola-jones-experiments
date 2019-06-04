from os.path import join, exists

import cv2
from numpy import concatenate, expand_dims
from pandas import read_csv, DataFrame

from config import WORKSPACE, SCALES, POSITIVE_PATH, MAX_POSITIVE_EXAMPLES
from utils.rectangle import _bbox_to_mask
from utils.scanner import scan_image

if __name__ == '__main__':

    dataframe_path = join(WORKSPACE, "images.csv")

    df = read_csv(dataframe_path, index_col=0)

    df = df.sample(frac=1.)

    assert isinstance(df, DataFrame)

    positive_samples_counter = 0

    for idx, elmt in df.iterrows():

        image_path = elmt.ImagePath

        image_arr = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

        bbox_mask = _bbox_to_mask(bbox=(elmt.Xmin, elmt.Xmax, elmt.Ymin, elmt.Ymax), shape=image_arr.shape[:2],
                                  burn_values=1)

        composite_image = concatenate([image_arr, expand_dims(bbox_mask, axis=-1)], axis=-1)

        bbox_area = (elmt.Xmax - elmt.Xmin) * (elmt.Ymax - elmt.Ymin)

        for scale in SCALES:

            for bb_idx, (bb, extim) in enumerate(
                    scan_image(
                        composite_image,
                        step=int(0.25 * scale),
                        size=scale
                    )):

                intersection = extim[:, :, -1].sum()

                ext_bb_area = scale ** 2

                iou = intersection / (bbox_area + ext_bb_area - intersection)

                if iou > 0.5:

                    positive_samples_counter += 1

                    pos_fn = join(POSITIVE_PATH, "ext_{0}_scale_{1}_{1:5d}.png".format(idx, scale, bb_idx))
                    if not exists(pos_fn):
                        cv2.imwrite(
                            pos_fn,
                            cv2.cvtColor(extim[:, :, :3], cv2.COLOR_RGB2BGR)
                        )

                if positive_samples_counter > MAX_POSITIVE_EXAMPLES:
                    break
            if positive_samples_counter > MAX_POSITIVE_EXAMPLES:
                break
        if positive_samples_counter > MAX_POSITIVE_EXAMPLES:
            break

            # imshow("Ext im", cv2.cvtColor(extim[:,:,:3], cv2.COLOR_RGB2BGR))
            # cv2.waitKey(10)
