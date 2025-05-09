import cv2
import random
import numpy as np
import albumentations as A

from ....file_utils.config_utils import read_configs

configs = read_configs()

PLANE_IDX = configs["PLANE_IDX"]
INPUT_SHAPE = configs["INPUT_SHAPE"]
TOLERANCE = configs["TOLERANCE"]


def scale(img, **kwargs):
    maximum = 1 if not img.max() else img.max()
    img = (img - img.min()) / (maximum - img.min())
    return img


def smoothen(img, **kwargs):
    img = np.where(img < TOLERANCE, 0, img)
    return img


class Augmentation:
    def __init__(self, plane="sagittal"):
        self.plane = plane

    def transformation_pipeline_(self):
        aug = A.Compose(
            [
                A.Sequential(
                    [
                        A.SomeOf(
                            [
                                A.Flip(p=1),
                                A.Rotate(
                                    180,
                                    interpolation=cv2.INTER_LANCZOS4,
                                    p=0.5,
                                    border_mode=cv2.BORDER_REFLECT,
                                    mask_value=0,
                                ),
                                A.SafeRotate(
                                    180,
                                    interpolation=cv2.INTER_LANCZOS4,
                                    p=1,
                                    border_mode=cv2.BORDER_REFLECT,
                                    mask_value=0,
                                ),
                                A.RandomResizedCrop(
                                    *INPUT_SHAPE[self.plane][:2],
                                    interpolation=cv2.INTER_LANCZOS4,
                                    p=0.5,
                                    scale=(0.85, 1.0),
                                    ratio=(0.9, 1.1)
                                ),
                                A.RandomBrightnessContrast(0.1, 0.1, p=0),
                                A.Affine(
                                    scale=(0.8, 1.4),
                                    translate_px=(0, 30),
                                    rotate=(-360, 360),
                                    shear=5,
                                    interpolation=cv2.INTER_LANCZOS4,
                                    mask_interpolation=0,
                                    cval=0,
                                    cval_mask=0,
                                    mode=0,
                                    keep_ratio=True,
                                    p=1,
                                ),
                                A.ElasticTransform(
                                    alpha=4,
                                    sigma=50,
                                    alpha_affine=25,
                                    interpolation=cv2.INTER_LANCZOS4,
                                    border_mode=cv2.BORDER_REFLECT,
                                    mask_value=0,
                                    approximate=True,
                                    same_dxdy=True,
                                    p=0.9,
                                ),
                                A.GridDistortion(
                                    num_steps=6,
                                    distort_limit=0.4,
                                    interpolation=4,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    value=0,
                                    mask_value=0,
                                    normalized=True,
                                    p=0.5,
                                ),
                                A.Perspective(
                                    interpolation=cv2.INTER_LANCZOS4,
                                    p=0.5,
                                    pad_mode=cv2.BORDER_CONSTANT,
                                    scale=[0.01, 0.1],
                                ),
                                A.ShiftScaleRotate(
                                    shift_limit=0.0625,
                                    scale_limit=[0.2, 0.2],
                                    rotate_limit=180,
                                    interpolation=4,
                                    border_mode=cv2.BORDER_REFLECT,
                                    mask_value=0,
                                    shift_limit_x=0.12,
                                    shift_limit_y=0.12,
                                    rotate_method="largest_box",
                                    always_apply=False,
                                    p=1,
                                ),
                            ],
                            replace=False,
                            n=3,
                            p=1,
                        ),
                        A.Lambda(name="scale_image", image=scale, mask=scale, p=1),
                    ],
                    p=1,
                ),
                A.Sequential(
                    [
                        A.OneOf(
                            [
                                A.GaussNoise(
                                    var_limit=(0.0001, 0.0003),
                                    mean=-0.1,
                                    per_channel=True,
                                    p=0.1,
                                ),
                                A.GaussNoise(
                                    var_limit=(0.0001, 0.0003),
                                    mean=0.1,
                                    per_channel=True,
                                    p=0.1,
                                ),
                                A.GaussNoise(
                                    var_limit=(0.0001, 0.0003),
                                    mean=0,
                                    per_channel=True,
                                    p=0.1,
                                ),
                            ],
                            p=0.2,
                        ),
                        A.Lambda(name="scale_image", image=scale, mask=scale, p=1),
                        A.Lambda(
                            name="smooth_image", image=smoothen, mask=smoothen, p=1
                        ),
                    ],
                    p=1,
                ),
            ]
        )
        return aug

    def augment_(self, img, msk):

        img_tr = np.rollaxis(img, PLANE_IDX[self.plane], 3)
        msk_tr = np.rollaxis(msk, PLANE_IDX[self.plane], 3)

        aug = self.transformation_pipeline_()
        augmented = aug(
            image=img_tr.astype(np.float32),
            mask=msk_tr.astype(np.float32),
        )

        image_aug = augmented["image"]
        mask_aug = augmented["mask"]

        augmented_imgs = np.rollaxis(image_aug, 2, 0)
        augmented_msks = np.rollaxis(mask_aug, 2, 0)

        return augmented_imgs, augmented_msks
