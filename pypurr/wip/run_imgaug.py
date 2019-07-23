import imgaug
from skimage.data import chelsea

if __name__ == '__main__':
    raw_im = chelsea()
    im = imgaug.imresize_single_image(raw_im, raw_im.shape[:2])

    # j,i,j+,i+
    bb = imgaug.BoundingBox(128, 64, 128 + 64 + 160, 64 + 128)

    imgaug.imshow(bb.draw_on_image(im, color=(0, 255, 0), size=5))

    seq = imgaug.augmenters.Sequential(
        [
            imgaug.augmenters.Resize((0.5,2.)),
            imgaug.augmenters.Fliplr(),
            imgaug.augmenters.Flipud(),
            imgaug.augmenters.CropToFixedSize(width=128, height=128)
        ]
    )

    for _ in range(10):
        aug_ims, aug_bbs = seq(image=im, bounding_boxes=imgaug.BoundingBoxesOnImage([bb], shape=im.shape))

        imgaug.imshow(
            aug_bbs.draw_on_image(aug_ims, color=(0, 255, 0), size=5)
        )
