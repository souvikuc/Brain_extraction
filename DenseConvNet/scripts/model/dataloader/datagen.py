import os, h5py, json
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from skimage.exposure import (
    equalize_adapthist,
    rescale_intensity,
)

from .utils import (
    crop_,
    conform_nii_,
    process_data_,
    Create_GT_labels,
    multiprocess_conform_,
)

# from ..core.model import ResidualDenseConvNet
from ....config.baseclass import BaseDataLoader
from ....file_utils.config_utils import read_configs

configs = read_configs()

LOG_DIR = configs["LOG_DIR"]
PLANE_IDX = configs["PLANE_IDX"]
BATCH_SIZE = configs["BATCH_SIZE"]
STEPS_PER_EPOCH = configs["STEPS_PER_EPOCH"]
TOLERANCE = configs["TOLERANCE"]


class NiftiGenerator_(Sequence):
    def __init__(
        self,
        directory,
        folder,
        batch_size=BATCH_SIZE,
        plane="sagittal",
        orientation="LAS",
    ):

        self.directory = directory
        self.folder = folder
        self.batch_size = batch_size
        self.plane = plane
        self.orientation = orientation
        self.data = DataLoader(
            self.directory,
            self.folder,
            self.batch_size,
            self.plane,
            self.orientation,
        )

    def __len__(self):
        return STEPS_PER_EPOCH

    def __getitem__(self, idx):
        (img_batch, laplace_batch, inv_map_batch), (
            msk_batch,
            boundary_batch,
        ) = self.data.preprocess_data_()

        return (img_batch, laplace_batch, inv_map_batch), (
            msk_batch,
            boundary_batch,
        )


class DataLoader(BaseDataLoader):
    def __init__(
        self,
        directory,
        folder,
        batch_size=BATCH_SIZE,
        plane="sagittal",
        orientation="LAS",
    ):

        self.directory = directory
        self.folder = folder
        self.batch_size = batch_size
        self.plane = plane
        self.orientation = orientation

    def load_data_(self):

        self.ng_dataset = Path(os.path.join(self.directory, self.folder))

        self.ng_subj_path = np.random.choice(sorted(list(self.ng_dataset.glob("R*"))))
        self.ng_subj_path = self.ng_subj_path / "brain_extraction"

        self.ng_brain_path = list(self.ng_subj_path.glob("T1w_acpc.nii.gz"))
        self.ng_brain_mask_path = list(
            self.ng_subj_path.glob("T1w_acpc_brain_mask.nii.gz")
        )

        self.input_data = self.ng_brain_path + self.ng_brain_mask_path
        conformed_data = multiprocess_conform_(self.input_data, self.orientation)

        return conformed_data

    def preprocess_data_(self):

        ng_brain, ng_brain_mask = self.load_data_()
        (
            (img_batch, laplace_batch, inv_map_batch),
            (mask_batch, boundary_batch),
        ) = process_data_(ng_brain, ng_brain_mask, self.plane)
        return (img_batch, laplace_batch, inv_map_batch), (mask_batch, boundary_batch)

    def get_generator_(self):
        gen = NiftiGenerator_(
            self.directory, self.folder, self.batch_size, self.plane, self.orientation
        )
        return gen

    def get_test_sample(self):

        folder = np.random.choice(["train", "test", "val"])
        self.dataset_path = Path(os.path.join(self.directory, folder))

        offset = 0.0

        self.subj_path = np.random.choice(sorted(list(self.dataset_path.glob("R*"))))
        self.subj_path = self.subj_path / "brain_extraction"
        self.brain_path = list(self.subj_path.glob("T1w_acpc.nii.gz"))

        img_np = conform_nii_(*self.brain_path, convention=self.orientation)
        img_np_rolled = np.rollaxis(img_np, PLANE_IDX[self.plane], 0)
        idx = np.random.choice(img_np_rolled.shape[0])

        sample = np.expand_dims(img_np_rolled[idx], axis=0)

        maximum = 1 if not sample.max() else sample.max()
        sample = (sample - sample.min()) / (maximum - sample.min())

        gt_labels = Create_GT_labels(sample, None)
        lp_map = gt_labels.create_lm_()

        sample = equalize_adapthist(sample) + offset
        sample = rescale_intensity(sample) + offset
        sample = np.where(sample < TOLERANCE, 0, sample)
        # sample = np.rollaxis(sample, PLANE_IDX[self.plane], 0)
        sample = np.expand_dims(sample, axis=-1)

        return [sample, lp_map]
