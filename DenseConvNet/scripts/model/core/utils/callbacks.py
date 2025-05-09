import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from ...dataloader.datagen import DataLoader
from .....file_utils.config_utils import read_configs

configs = read_configs()
DATA_DIR = configs["DATA_DIR"]
LOG_DIR = configs["LOG_DIR"]


class TrainTimeCallback(Callback):
    def __init__(self, plane="sagittal"):

        assert plane in [
            "axial",
            "coronal",
            "sagittal",
        ], f"plane must be one of sagittal, coronal, or axial, given {plane}"

        self.plane = plane
        self.time_started = None
        self.time_finished = None

    def on_train_begin(self, logs=None):
        self.time_started = datetime.now()
        print(f"\n\n{'-'*51}")
        print(
            f"|  {self.plane.upper()}-TRAINING STARTED : {self.time_started.strftime('%Y-%m-%d  %H:%M')}  |"
        )
        print(f"{'-'*51}\n\n")

    def on_train_end(self, logs=None):
        self.time_finished = datetime.now()
        train_duration = str(self.time_finished - self.time_started)
        print(f"\n\n{'-'*100}")
        print(
            f"\n\n{self.plane.upper()}-TRAINING FINISHED : {self.time_finished.strftime('%Y-%m-%d  %H:%M')}  ||  RUNTIME : {train_duration}\n\n"
        )
        print(f"{'-'*100}\n\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.time_curr_epoch = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = str(datetime.now() - self.time_curr_epoch)

        tml, vml = logs["mean_loss"], logs["val_mean_loss"]
        ttml, vtml = logs["total_mean_loss"], logs["val_total_mean_loss"]

        tdl_seg, vdl_seg = (
            logs["segmentation_dice_loss"],
            logs["val_segmentation_dice_loss"],
        )
        tds_seg, vds_seg = (
            logs["segmentation_dice_similarity"],
            logs["val_segmentation_dice_similarity"],
        )
        tsds_seg, vsds_seg = (
            logs["segmentation_selected_dice_similarity"],
            logs["val_segmentation_selected_dice_similarity"],
        )
        tads_seg, vads_seg = (
            logs["segmentation_avg_dice_similarity"],
            logs["val_segmentation_avg_dice_similarity"],
        )

        tdl_fusion, vdl_fusion = (
            logs["fusion_dice_loss"],
            logs["val_fusion_dice_loss"],
        )
        tds_fusion, vds_fusion = (
            logs["fusion_dice_similarity"],
            logs["val_fusion_dice_similarity"],
        )
        tsds_fusion, vsds_fusion = (
            logs["fusion_selected_dice_similarity"],
            logs["val_fusion_selected_dice_similarity"],
        )
        tads_fusion, vads_fusion = (
            logs["fusion_avg_dice_similarity"],
            logs["val_fusion_avg_dice_similarity"],
        )

        tcel_final, vcel_final = (
            logs["final_cross_entropy_loss"],
            logs["val_final_cross_entropy_loss"],
        )
        tfl_final, vfl_final = (
            logs["final_focal_loss"],
            logs["val_final_focal_loss"],
        )

        tpcel_final, vpcel_final = (
            logs["final_pixelwise_cross_entropy_loss"],
            logs["val_final_pixelwise_cross_entropy_loss"],
        )
        tpfl_final, vpfl_final = (
            logs["final_pixelwise_focal_loss"],
            logs["val_final_pixelwise_focal_loss"],
        )

        tsds_final, vsds_final = (
            logs["final_segmentation_dice_similarity"],
            logs["val_final_segmentation_dice_similarity"],
        )
        tssds_final, vssds_final = (
            logs["final_segmentation_selected_dice_similarity"],
            logs["val_final_segmentation_selected_dice_similarity"],
        )
        tads_final, vads_final = (
            logs["final_avg_dice_similarity"],
            logs["val_final_avg_dice_similarity"],
        )

        tbds_final, vbds_final = (
            logs["final_boundary_dice_similarity"],
            logs["val_final_boundary_dice_similarity"],
        )
        tbsds_final, vbsds_final = (
            logs["final_boundary_selected_dice_similarity"],
            logs["val_final_boundary_selected_dice_similarity"],
        )
        tbads_final, vbads_final = (
            logs["final_boundary_avg_dice_similarity"],
            logs["val_final_boundary_avg_dice_similarity"],
        )

        tcel_bound, vcel_bound = (
            logs["boundary_cross_entropy_loss"],
            logs["val_boundary_cross_entropy_loss"],
        )
        tfl_bound, vfl_bound = (
            logs["boundary_focal_loss"],
            logs["val_boundary_focal_loss"],
        )
        metrics_ = f"train_total_mean_loss:                         {ttml:.5f}, {'-'*10}  valid_total_mean_loss:                         {vtml:.5f}\n"
        metrics1 = f"train_mean_loss:                               {tml:.5f}, {'-'*10} valid_mean_loss:                               {vml:.5f}\n"

        metrics1_seg = f"train_seg_dice_loss:                           {tdl_seg:.5f}, {'-'*10} valid_seg_dice_loss:                           {vdl_seg:.5f}\n"
        metrics2_seg = f"train_seg_dice_similarity:                     {tds_seg:.5f}, {'-'*10} valid_seg_dice_similarity:                     {vds_seg:.5f}\n"
        metrics3_seg = f"train_seg_selected_dice_similarity:            {tsds_seg:.5f}, {'-'*10} valid_seg_selected_dice_similarity:            {vsds_seg:.5f}\n"
        metrics4_seg = f"train_seg_avg_dice_similarity:                 {tads_seg:.5f}, {'-'*10} valid_seg_avg_dice_similarity:                 {vads_seg:.5f}\n\n"

        metrics1_fusion = f"train_fusion_dice_loss:                        {tdl_fusion:.5f}, {'-'*10} valid_fusion_dice_loss:                        {vdl_fusion:.5f}\n"
        metrics2_fusion = f"train_fusion_dice_sim   ilarity:               {tds_fusion:.5f}, {'-'*10} valid_fusion_dice_similarity:                  {vds_fusion:.5f}\n"
        metrics3_fusion = f"train_fusion_selected_dice_similarity:         {tsds_fusion:.5f}, {'-'*10} valid_fusion_selected_dice_similarity:         {vsds_fusion:.5f}\n"
        metrics4_fusion = f"train_fusion_avg_dice_similarity:              {tads_fusion:.5f}, {'-'*10} valid_fusion_avg_dice_similarity:              {vads_fusion:.5f}\n\n"

        metrics1_final = f"train_final_cross_entropy_loss:                {tcel_final:.5f}, {'-'*10} valid_final_cross_entropy_loss:                {vcel_final:.5f}\n"
        metrics2_final = f"train_final_pixelwise_cross_entropy_loss:      {tpcel_final:.5f}, {'-'*10} valid_final_pixelwise_cross_entropy_loss:      {vpcel_final:.5f}\n"
        metrics3_final = f"train_final_focal_loss:                        {tfl_final:.5f}, {'-'*10} valid_final_focal_loss:                        {vfl_final:.5f}\n"
        metrics4_final = f"train_final_pixelwise_focal_loss:              {tpfl_final:.5f}, {'-'*10} valid_final_pixelwise_focal_loss:              {vpfl_final:.5f}\n"
        metrics5_final = f"train_final_segmentation_dice_similarity:      {tsds_final:.5f}, {'-'*10} valid_final_segmentation_dice_similarity:      {vsds_final:.5f}\n"
        metrics6_final = f"train_final_selected_dice_similarity:          {tssds_final:.5f}, {'-'*10} valid_final_selected_dice_similarity:          {vssds_final:.5f}\n"
        metrics7_final = f"train_final_avg_dice_similarity:               {tads_final:.5f}, {'-'*10} valid_final_avg_dice_similarity:               {vads_final:.5f}\n"
        metrics8_final = f"train_final_boundary_dice_similarity:          {tbds_final:.5f}, {'-'*10} valid_final_boundary_dice_similarity:          {vbds_final:.5f}\n"
        metrics9_final = f"train_final_boundary_selected_dice_similarity: {tbsds_final:.5f}, {'-'*10} valid_final_boundary_selected_dice_similarity: {vbsds_final:.5f}\n"
        metrics10_final = f"train_final_boundary_avg_dice_similarity:      {tbads_final:.5f}, {'-'*10} valid_final_boundary_avg_dice_similarity:      {vbads_final:.5f}\n"

        metrics1_bound = f"train_boundary_cross_entropy_loss:             {tcel_bound:.5f}, {'-'*10} valid_final_boundary_cross_entropy_loss:       {vcel_bound:.5f}\n"
        metrics2_bound = f"train_boundary_focal_loss:                     {tfl_bound:.5f}, {'-'*10} valid_final_boundary_focal_loss:               {vfl_bound:.5f}\n"

        print()
        print("-" * 42)
        print(f"\n\n | Epoch: {(epoch+1):4}  ||  Runtime: {epoch_duration} | >>")
        print("-" * 42, "\n")
        print(f"   {metrics_}   {metrics1}")

        print("SEGMENTATION BRANCH:")
        print("--------------------")
        print(f"   {metrics1_seg}   {metrics2_seg}   {metrics3_seg}   {metrics4_seg}")

        print("BOUNDARY BRANCH:")
        print("--------------------")
        print(f"   {metrics1_bound}   {metrics2_bound}")

        print("FUSION BRANCH:")
        print("--------------------")
        print(
            f"   {metrics1_fusion}   {metrics2_fusion}   {metrics3_fusion}   {metrics4_fusion}"
        )

        print("OUTPUT BRANCH:")
        print("--------------------")
        print(
            f"   {metrics1_final}   {metrics2_final}   {metrics3_final}   {metrics4_final}   {metrics5_final}   {metrics6_final}   {metrics7_final}   {metrics8_final}   {metrics9_final}   {metrics10_final}"
        )


class CustomEarlyStopping(Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_v_loss = np.Inf
        self.best_mean_dl = np.Inf
        self.best_mean_pfl = np.Inf
        self.best_mean_ds = 0

    def on_epoch_end(self, epoch, logs=None):
        v_loss = logs.get("val_mean_loss")

        vsdl_seg = logs.get("val_segmentation_dice_loss")
        vsdl_fusion = logs.get("val_fusion_dice_loss")
        mean_dl = (vsdl_seg + vsdl_fusion) / 2

        vbfl_bound = logs.get("val_boundary_focal_loss")
        vpfl_final = logs.get("val_final_pixelwise_focal_loss")
        mean_pfl = (vbfl_bound + vpfl_final) / 2

        vacds_seg = logs.get("val_segmentation_dice_similarity")
        vacds_fusion = logs.get("val_fusion_dice_similarity")
        vacds_final = logs.get("val_final_segmentation_dice_similarity")
        mean_ds = (vacds_seg + vacds_fusion + vacds_final) / 3

        # If ALL the following conditions do not improve for 'patience' epochs, stop training early.

        condition1 = np.less_equal(v_loss, self.best_v_loss)
        condition2 = np.less_equal(mean_dl, self.best_mean_dl)
        condition3 = np.less_equal(mean_pfl, self.best_mean_pfl)
        condition4 = np.greater_equal(mean_ds, self.best_mean_ds)

        if condition1 and condition2 and condition3 and condition4:

            self.best_v_loss = v_loss
            self.best_mean_dl = mean_dl
            self.best_mean_pfl = mean_pfl
            self.best_mean_ds = mean_ds
            if self.wait != 0:
                print(f"Wait was {self.wait}. Reset to 0......\n")
                self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            print(f"Wait increased by 1. Current wait is {self.wait}...\n")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("\n", "-" * 30)
            print("Epoch %03d: early stopping" % (self.stopped_epoch + 1))


class SavePrediction(Callback):
    def __init__(self, plane="sagittal"):
        self.plane = plane
        self.folder = np.random.choice(["train", "test", "val"])
        self.test_gen = DataLoader(
            DATA_DIR, self.folder, plane=self.plane, batch_size=1
        )
        self.test_data = self.test_gen.get_test_sample()

    def on_epoch_end(self, epoch, logs=None):

        y_pred = self.model.predict(self.test_data)

        y_orig = self.test_data[0][0]
        y_boundary_pred = y_pred["final_boundary_mask"][0]
        y_mask_pred = y_pred["final_segmentation_mask"][0]

        lists = [y_orig, y_boundary_pred, y_mask_pred]
        list_names = ["input", "predicted boundary", "predicted segmentation"]

        fig, axs = plt.subplots(1, 3, figsize=(10, 12))
        for i, ax in enumerate(axs):
            ax.imshow(lists[i], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{list_names[i]}")

        plt.tight_layout()
        plt.savefig(
            Path(LOG_DIR) / f"predictions/{self.plane}/prediction_{epoch+1}.png"
        )
        plt.close(fig)

        # idx = np.random.choice(np.arange(40, 200))
        # tf.keras.utils.save_img(
        #     Path(LOG_DIR) / f"predictions/{self.plane}/seg_prediction_{epoch+1}.png",
        #     y_pred["final_segmentation_mask"][0],
        #     scale=True,
        # )
        # tf.keras.utils.save_img(
        #     Path(LOG_DIR)
        #     / f"predictions/{self.plane}/boundary_prediction_{epoch+1}.png",
        #     y_pred["final_boundary_mask"][0],
        #     scale=True,
        # )
