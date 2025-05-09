import numpy as np
import nibabel as nib
import tensorflow as tf
import multiprocessing as mp
from joblib import Parallel, delayed
from nibabel.processing import conform
from nibabel.orientations import aff2axcodes
from skimage.morphology import diameter_closing
from scipy.ndimage import distance_transform_edt
from nilearn.image import resample_img
from skimage.exposure import (
    equalize_adapthist,
    rescale_intensity,
)
from skimage.filters import threshold_local
import cv2


from .augment import Augmentation
from ....file_utils.config_utils import read_configs

configs = read_configs()

PLANE_IDX = configs["PLANE_IDX"]
COMP_PLANE_IDX = configs["COMP_PLANE_IDX"]
NUM_CLASS = configs["NUM_CLASS"]
BATCH_SIZE = configs["BATCH_SIZE"]

TOLERANCE = configs["TOLERANCE"]
# VOXEL_SIZE = configs["VOXEL_SIZE"]
VOLUME_SHAPE = configs["VOLUME_SHAPE"]

# ----------------------------------------------------------------
# functions for conforming input nifti files
# ----------------------------------------------------------------


def is_conform_(img, convention="LAS"):
    """returns bool depending on whether the input file is in the mentioned convention.
    img: nifti object file
    convention: Any convention e.g. LAS, RAS etc
    """
    try:
        orientation = aff2axcodes(img.affine)
    except TypeError:
        raise TypeError("img must be a nifti object with 'affine' attribute")

    if "".join(orientation) == convention:
        return True
    else:
        return False


def conform_nii_(img_path, convention="LAS", out_shape=tuple(VOLUME_SHAPE)):
    """conform the input file to the mentioned convention.
    img_path: path-like object of the nifti object file
    convention: Any convention e.g. LAS, RAS etc
    """
    try:
        img = nib.load(img_path)
    except TypeError:
        raise TypeError("img_path must be a path to a nifti object")

    if not is_conform_(img, convention):
        img = conform(
            img,
            orientation=convention,
            out_shape=out_shape,
            voxel_size=img.header.get_zooms(),
        )
    else:
        img = resample_img(img, target_affine=img.affine, target_shape=out_shape)
    img_np = img.get_fdata()

    return img_np


def multiprocess_conform_(img_paths, convention="LAS"):
    """multiprocess function to conform the input file to the mentioned convention.
    img_paths: path-like objects of the nifti object files
    convention: Any convention e.g. LAS, RAS etc
    """
    conventions = [convention] * len(img_paths)
    lists = [(img_paths[i], conventions[i]) for i in range(len(img_paths))]

    num_cores = 10

    results = Parallel(num_cores, backend="threading")(
        delayed(conform_nii_)(*args) for args in lists
    )
    return results


# ----------------------------------------------------------------
# functions for multiprocessing input images
# ----------------------------------------------------------------


def process_data_(image, mask, plane):

    # t1 = TOLERANCE
    maximum = 1 if not image.max() else image.max()
    offset = 0.0

    brain_final = (image - image.min()) / (maximum - image.min())
    # brain_final = np.pad(brain_final, ((0, 0), (0, 1), (0, 0)))
    # brain_final = brain_final[2:-2, :, 2:-2]
    brain_final = equalize_adapthist(brain_final) + offset
    brain_final = rescale_intensity(brain_final) + offset
    brain_final = np.where(brain_final < TOLERANCE, 0, brain_final)

    # brain_mask_final = np.pad(mask, ((0, 0), (0, 1), (0, 0)))
    # brain_mask_final = brain_mask_final[2:-2, :, 2:-2]
    brain_mask_final = np.where(mask < 1.0, 0.0, mask)

    brain_final_cropped, brain_mask_final_cropped = crop_(
        brain_final, brain_mask_final, plane=plane, tol=0.25
    )
    np.random.seed(12)
    slice_idx = np.random.choice(
        brain_final_cropped.shape[PLANE_IDX[plane]],
        int(BATCH_SIZE / 2),
        replace=False,
    )

    img_batch = brain_final_cropped.take(slice_idx, PLANE_IDX[plane])
    brain_msk_batch = brain_mask_final_cropped.take(slice_idx, PLANE_IDX[plane])

    img_batch_arranged = np.rollaxis(img_batch, PLANE_IDX[plane], 0)
    brain_msk_batch_arranged = np.rollaxis(
        brain_msk_batch,
        PLANE_IDX[plane],
        0,
    )

    aug = Augmentation(plane=plane)
    aug_img_batch, aug_brain_msk_batch = aug.augment_(img_batch, brain_msk_batch)
    aug_img_batch = np.where(aug_img_batch < TOLERANCE, 0, aug_img_batch)

    concated_image_batch = np.concatenate((img_batch_arranged, aug_img_batch))
    concated_mask_batch = np.concatenate(
        (brain_msk_batch_arranged, aug_brain_msk_batch)
    )

    groundtruth_label = Create_GT_labels(concated_image_batch, concated_mask_batch)
    (final_img_batch, final_laplace_map_batch, inv_dist_map_batch), (
        final_msk_batch,
        final_boundary_batch,
    ) = groundtruth_label.generate_input_data_()

    return (final_img_batch, final_laplace_map_batch, inv_dist_map_batch), (
        final_msk_batch,
        final_boundary_batch,
    )


# ----------------------------------------------------------------
# class for creating masks, boundaries and distance transform maps
# ----------------------------------------------------------------


class Create_GT_labels:
    def __init__(self, brain, brain_mask):
        self.brain = brain

        self.brain_mask = brain_mask

    def create_masks_(self, tol=TOLERANCE + 0.1):
        skull = np.where(self.brain_mask == 1, 0.0, self.brain)
        skull_cond = np.logical_and(self.brain_mask != 1, skull > tol)
        skull = skull_cond.astype(np.float32)
        skull = np.stack(
            [
                diameter_closing(skull[i], diameter_threshold=30)
                for i in range(skull.shape[0])
            ]
        )
        mask = skull + self.brain_mask * 2
        mask = np.where(mask == 3, 2, mask)
        return mask

    def create_boundary_contour_(self, masks):
        # mask = self.create_masks_()

        # creating background boundary
        bg_mask = tf.expand_dims(tf.where(masks == 0, 1.0, 0.0), -1)
        bg_min_pool = tf.nn.max_pool2d(-bg_mask, 3, 1, "SAME") * -1
        bg_max_min_pool = tf.nn.max_pool2d(bg_min_pool, 3, 1, "SAME")
        bg_boundary = tf.nn.relu(bg_max_min_pool - bg_min_pool)

        # creating skull boundary
        s_mask = tf.expand_dims(tf.where(masks == 1, 1.0, 0.0), -1)
        s_min_pool = tf.nn.max_pool2d(-s_mask, 3, 1, "SAME") * -1
        s_max_min_pool = tf.nn.max_pool2d(s_min_pool, 3, 1, "SAME")
        s_boundary = tf.nn.relu(s_max_min_pool - s_min_pool)

        # creating brain boundary
        br_mask = tf.expand_dims(tf.where(masks == 2, 1.0, 0.0), -1)
        br_min_pool = tf.nn.max_pool2d(-br_mask, 3, 1, "SAME") * -1
        br_max_min_pool = tf.nn.max_pool2d(br_min_pool, 3, 1, "SAME")
        br_boundary = tf.nn.relu(br_max_min_pool - br_min_pool)

        return s_boundary + br_boundary + bg_boundary
        # return np.where(skull_boundary+brain_boundary !=0,1,0)

    def create_distance_maps_(self, mask):

        msk_onehot = tf.one_hot(mask, NUM_CLASS)
        msk_unstacked = np.split(msk_onehot, msk_onehot.shape[-1], axis=-1)
        dtm = [distance_transform_edt(i) for i in msk_unstacked]

        inv_dtm = [
            np.where(dist_map > 0, dist_map.max() - dist_map, 0) for dist_map in dtm
        ]

        dtm_stacked = np.concatenate(dtm, axis=-1)
        max_dtm = np.amax(dtm_stacked, axis=(0, 1), keepdims=True, initial=1.0)
        dtm_stacked = dtm_stacked / max_dtm

        inv_dtm_stacked = np.concatenate(inv_dtm, axis=-1)
        max_inv_dtm = np.amax(inv_dtm_stacked, axis=(0, 1), keepdims=True, initial=1.0)
        inv_dtm_stacked = inv_dtm_stacked / max_inv_dtm

        return inv_dtm_stacked

    def create_laplace_map_(self, img):

        ksizes = (3, 5)

        laplace_maps = [
            np.abs(cv2.Laplacian(img, ddepth=-1, ksize=ksize)) for ksize in ksizes
        ]
        laplace_maps = [
            (laplace_map - laplace_map.min()) / (laplace_map.max() - laplace_map.min())
            for laplace_map in laplace_maps
        ]

        laplace_maps_th = [
            threshold_local(laplace_map, block_size=ksize)
            for laplace_map, ksize in zip(laplace_maps, ksizes)
        ]

        summed_laplace_map = np.sum(np.stack(laplace_maps_th), axis=0)

        maxes = np.amax(summed_laplace_map)
        mins = np.amin(summed_laplace_map)
        laplace_map_final = (summed_laplace_map - mins) / (maxes - mins)

        return laplace_map_final

    def create_lm_(self):

        brain_unstacked = np.split(self.brain, self.brain.shape[0], axis=0)
        brain_unstacked = [np.squeeze(brain, axis=0) for brain in brain_unstacked]
        num_cores = 8
        results = Parallel(num_cores, backend="threading")(
            delayed(self.create_laplace_map_)(args) for args in brain_unstacked
        )

        laplace_map = np.expand_dims(np.stack(results, axis=0), -1)

        return laplace_map

    def create_dtm_(self, masks):

        masks_unstacked = np.split(masks, masks.shape[0], axis=0)
        # print("xxxxx", masks_unstacked[0].shape)
        masks_unstacked = [np.squeeze(msk, axis=0) for msk in masks_unstacked]
        num_cores = 8
        results = Parallel(num_cores, backend="threading")(
            delayed(self.create_distance_maps_)(args) for args in masks_unstacked
        )

        inv_dtm = np.stack(results, axis=0)

        return inv_dtm  # , dtm

    def generate_input_data_(self, tol=TOLERANCE + 0.0):
        final_mask = self.create_masks_(tol=tol)
        final_boundary_mask = self.create_boundary_contour_(final_mask)
        final_inv_dtm = self.create_dtm_(final_mask)
        final_laplace_map = self.create_lm_()

        final_brain = np.expand_dims(self.brain, -1)
        final_mask = np.expand_dims(final_mask, -1)

        return (final_brain, final_laplace_map, final_inv_dtm), (
            final_mask,
            final_boundary_mask,
        )


# ----------------------------------------------------------------
# functions for cropping and padding input images
# ----------------------------------------------------------------


def crop_(img, mask, plane="sagittal", tol=0.25):
    """crops the edge of the file with only background.
    img: img to crop
    mask: corresponding mask to crop
    plane: determines which sides of the 3D-volume to crop keeping the input image files of same shape
    tol: tolerance of cropping boundary
    """
    assert plane in PLANE_IDX.keys(), "plane must be sagittal, coronal or axial"

    thresh = img > tol
    shape = img.shape
    thresholded = thresh.any(tuple(COMP_PLANE_IDX[plane]))

    if plane == "sagittal":
        portion = (thresholded, range(shape[1]), range(shape[2]))
        img_cr, msk_cr = img[np.ix_(*portion)], mask[np.ix_(*portion)]
        return img_cr, msk_cr
    if plane == "coronal":
        portion = (range(shape[0]), thresholded, range(shape[2]))
        img_cr, msk_cr = img[np.ix_(*portion)], mask[np.ix_(*portion)]
        return img_cr, msk_cr
    if plane == "axial":
        portion = (range(shape[0]), range(shape[1]), thresholded)
        img_cr, msk_cr = img[np.ix_(*portion)], mask[np.ix_(*portion)]
        return img_cr, msk_cr
