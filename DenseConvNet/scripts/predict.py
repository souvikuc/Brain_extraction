import numpy as np
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf
import pathlib, json
import nibabel as nib
import multiprocessing as mp
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from nibabel.processing import conform
from skimage.exposure import (
    equalize_adapthist,
    rescale_intensity,
)

from .model.core.model import ResidualDenseConvNet
from ..file_utils.config_utils import read_configs

configs = read_configs()

PLANE_IDX = configs["PLANE_IDX"]
DATA_DIR = configs["DATA_DIR"]
LOG_DIR = configs["LOG_DIR"]
COMP_PLANE_IDX = configs["COMP_PLANE_IDX"]
TOLERANCE = configs["TOLERANCE"]


def get_weights(plane="sagittal", vol_shape=(256, 312, 256), minimum=-20, maximum=20):
    def sigmoid(z):
        return 1 / (1 + np.exp(-2 * z + 20))

    shape = tuple([vol_shape[i] for i in COMP_PLANE_IDX[plane]])

    resmin = np.ones(shape, dtype=np.float32) * minimum
    resmax = np.ones(shape, dtype=np.float32) * maximum
    values = sigmoid(
        np.linspace(
            resmin, resmax, int(vol_shape[PLANE_IDX[plane]] // 2), axis=PLANE_IDX[plane]
        )
    )
    i_values = sigmoid(
        np.linspace(
            resmax, resmin, int(vol_shape[PLANE_IDX[plane]] // 2), axis=PLANE_IDX[plane]
        )
    )
    return np.expand_dims(np.concatenate((values, i_values), axis=PLANE_IDX[plane]), -1)


class MakePredictions:
    def __init__(self, image_path, output_path):
        self.image_path = str(image_path)
        self.output_path = str(output_path)

    def process_data(self, img_np):
        offset = 0.0  # if "Neurogazer" in self.image_path else 0.04
        # threshold = 0.2

        maximum = 1 if not img_np.max() else img_np.max()
        img_scaled = (img_np - img_np.min()) / (maximum - img_np.min())
        img_np1 = equalize_adapthist(img_scaled) + offset
        img_np1 = rescale_intensity(img_np1) + offset
        img_np1 = np.where(img_np1 < TOLERANCE, 0, img_np1)

        return np.expand_dims(img_np1, -1)

    def load_data(self):

        img = nib.load(self.image_path)
        self.img_hdr, self.img_affn = img.header, img.affine
        img_cnf = conform(
            img,
            orientation="LAS",
            out_shape=(256, 312, 256),
            voxel_size=(0.7, 0.7, 0.7),
        )
        self.img_cnf_hdr, self.img_cnf_affn = img_cnf.header, img_cnf.affine
        img_cnf_np = img_cnf.get_fdata()
        img_np1 = self.process_data(img_cnf_np)
        img_np2 = np.rollaxis(img_np1, 1, 0)
        img_np3 = np.rollaxis(img_np1, 2, 0)

        # msk = nib.load(msk_path)
        # msk_cnf = conform_nii_(msk_path, convention="LAS", out_shape=(256, 312, 256))
        # msk_np = np.where(msk_np < 1.0, 0.0, msk_np)
        # msk_np = np.expand_dims(msk_np, -1)

        return img_np1, img_np2, img_np3

    def load_models(self):
        sagittal_model, coronal_model, axial_model = (
            ResidualDenseConvNet(plane="sagittal"),
            ResidualDenseConvNet(plane="coronal"),
            ResidualDenseConvNet(plane="axial"),
        )

        sagittal_model.model.load_weights(LOG_DIR + "/trained_weights_sagittal.h5")
        coronal_model.model.load_weights(LOG_DIR + "/trained_weights_coronal.h5")
        axial_model.model.load_weights(LOG_DIR + "/trained_weights_axial.h5")

        return sagittal_model.model, coronal_model.model, axial_model.model

    def predict(self, save=False):
        models = self.load_models()
        data = self.load_data()
        planes = sorted(list(PLANE_IDX.keys()), reverse=True)

        model_outputs = [
            m.predict(datum, verbose="auto") for m, datum in zip(models, data)
        ]
        probability_maps = [
            np.rollaxis(model_out["final_segmentation_map"], 0, PLANE_IDX[plane] + 1)
            for model_out, plane in zip(model_outputs, planes)
        ]

        # weights = [
        #     get_weights(plane="sagittal"),
        #     get_weights(plane="coronal"),
        #     get_weights(plane="axial"),
        # ]
        # print([w.shape for w in weights])
        # weights = [
        #     np.rollaxis(weight, 0, PLANE_IDX[plane] + 1)
        #     for weight, plane in zip(weights, planes)
        # ]

        # weighted_prob_map = [
        #     prob_map * weight for weight, prob_map in zip(weights, probability_maps)
        # ]

        combined_prob_map = np.sum(np.stack(probability_maps), axis=0) / 3
        # combined_w_prob_map = np.sum(np.stack(weighted_prob_map), axis=0)

        final_mask1 = np.argmax(combined_prob_map, -1)
        # final_mask2 = np.argmax(combined_w_prob_map, -1)

        if save:
            # self.save_nii(final_mask1 == 2)
            self.save_nii(final_mask1 == 2)

        return (
            data[0],
            combined_prob_map,
            # combined_w_prob_map,
            final_mask1,
            # final_mask2,
        )

    def save_nii(self, result):
        segmentation = nib.Nifti2Image(
            result, affine=self.img_cnf_affn, header=self.img_cnf_hdr
        )
        nib.save(segmentation, self.output_path)
