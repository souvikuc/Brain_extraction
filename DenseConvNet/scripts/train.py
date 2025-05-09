from silence_tensorflow import silence_tensorflow
import os, sys
import pdb
import time

from pathlib import Path
import tensorflow as tf
from datetime import datetime
from contextlib import redirect_stdout
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
)

from ..file_utils.config_utils import read_configs

# from ..file_utils.helper_functions import create_output_folders

configs = read_configs()

EPOCHS = configs["EPOCHS"]
LOG_DIR = configs["LOG_DIR"]
EPSILON = configs["EPSILON"]
DATA_DIR = configs["DATA_DIR"]
MODEL_DIR = configs["MODEL_DIR"]
NUM_CLASS = configs["NUM_CLASS"]
BATCH_SIZE = configs["BATCH_SIZE"]
STEPS_PER_EPOCH = configs["STEPS_PER_EPOCH"]

# ------------------------------------------------------------------------------
# Starting the training...
# ------------------------------------------------------------------------------
from .model.core.utils.callbacks import (
    TrainTimeCallback,
    CustomEarlyStopping,
    SavePrediction,
)
from .model.core.model import ResidualDenseConvNet


class Train_Model:
    def __init__(self, dataset_path, config_file_path, output_path):

        self.dataset_path = Path(dataset_path)
        self.config_file_path = Path(config_file_path)
        if output_path == None:
            # self.output_path = self.config_file_path.parent.parent.parent
            self.output_path = sys.path[0]
        else:
            self.output_path = Path(output_path)

        # set_Config_folders(self.dataset_path, self.config_file_path, self.output_path)

    def load_callbacks(self, plane):
        if plane == "sagittal":
            sagittal_weight_filename = f"/trained_weights_sagittal.h5"
            sagittal_result_filename = f"results_sagittal.csv"

            checkpoint_sagittal = ModelCheckpoint(
                LOG_DIR + sagittal_weight_filename,
                monitor="val_final_segmentation_dice_similarity",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
            )
            savepred_sagittal = SavePrediction(plane="sagittal")
            train_time_sagittal = TrainTimeCallback(plane="sagittal")
            earlystop_sagittal = CustomEarlyStopping(patience=16)
            csv_log_sagittal = CSVLogger(
                os.path.join(LOG_DIR, sagittal_result_filename)
            )
            reduce_lr_sagittal = ReduceLROnPlateau(
                monitor="val_mean_loss",
                factor=0.98,
                mode="min",
                min_delta=0.000000001,
                patience=4,
            )
            return {
                "model_checkpoint": checkpoint_sagittal,
                "per_epoch_predictions": savepred_sagittal,
                "training_time": train_time_sagittal,
                "earlystopping": earlystop_sagittal,
                "csv_logger": csv_log_sagittal,
                "leaning_rate": reduce_lr_sagittal,
            }

        elif plane == "coronal":
            coronal_weight_filename = f"/trained_weights_coronal.h5"
            coronal_result_filename = f"results_coronal.csv"

            checkpoint_coronal = ModelCheckpoint(
                LOG_DIR + coronal_weight_filename,
                monitor="val_final_segmentation_dice_similarity",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
            )

            savepred_coronal = SavePrediction(plane="coronal")
            train_time_coronal = TrainTimeCallback(plane="coronal")
            earlystop_coronal = CustomEarlyStopping(patience=16)
            csv_log_coronal = CSVLogger(os.path.join(LOG_DIR, coronal_result_filename))
            reduce_lr_coronal = ReduceLROnPlateau(
                monitor="val_mean_loss",
                factor=0.98,
                mode="min",
                min_delta=0.000000001,
                patience=4,
            )
            return {
                "model_checkpoint": checkpoint_coronal,
                "per_epoch_predictions": savepred_coronal,
                "training_time": train_time_coronal,
                "earlystopping": earlystop_coronal,
                "csv_logger": csv_log_coronal,
                "leaning_rate": reduce_lr_coronal,
            }

        elif plane == "axial":
            axial_weight_filename = f"/trained_weights_axial.h5"
            axial_result_filename = f"results_axial.csv"

            checkpoint_axial = ModelCheckpoint(
                LOG_DIR + axial_weight_filename,
                monitor="val_final_segmentation_dice_similarity",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
            )

            savepred_axial = SavePrediction(plane="axial")
            train_time_axial = TrainTimeCallback(plane="axial")
            earlystop_axial = CustomEarlyStopping(patience=16)
            csv_log_axial = CSVLogger(os.path.join(LOG_DIR, axial_result_filename))
            reduce_lr_axial = ReduceLROnPlateau(
                monitor="val_mean_loss",
                factor=0.98,
                mode="min",
                min_delta=0.000000001,
                patience=4,
            )
            return {
                "model_checkpoint": checkpoint_axial,
                "per_epoch_predictions": savepred_axial,
                "training_time": train_time_axial,
                "earlystopping": earlystop_axial,
                "csv_logger": csv_log_axial,
                "leaning_rate": reduce_lr_axial,
            }

        else:
            raise ValueError("The plane must be one of 'sagittal','coronal' or 'axial'")

    def load_model_and_data(self, plane):
        if plane == "sagittal":

            s_model_file = f"/sagittal_model.png"
            s_model_summary = f"/sagittal_model_summary.txt"

            sagittal_model = ResidualDenseConvNet(
                "sagittal", num_cls=NUM_CLASS, epsilon=EPSILON
            )
            sagittal_model.model.plot_model_(
                show_shapes=True,
                show_dtype=True,
                dpi=140,
                show_layer_activations=True,
                to_file=MODEL_DIR + s_model_file,
            )
            with open(MODEL_DIR + s_model_summary, "w") as f:
                with redirect_stdout(f):
                    sagittal_model.model.summary(
                        show_trainable=True, expand_nested=True
                    )
                sagittal_model.model.summary(
                            show_trainable=True, expand_nested=True
                        )

            train_data_sagittal = sagittal_model.load_generator_(
                DATA_DIR, "train", batch_size=BATCH_SIZE, orientation="LAS"
            )
            validation_data_sagittal = sagittal_model.load_generator_(
                DATA_DIR, "val", batch_size=BATCH_SIZE, orientation="LAS"
            )

            return {
                "model": sagittal_model.model,
                "train_data": train_data_sagittal,
                "validation_data": validation_data_sagittal,
            }

        elif plane == "coronal":

            c_model_file = f"/coronal_model.png"
            c_model_summary = f"/coronal_model_summary.txt"

            coronal_model = ResidualDenseConvNet(
                "coronal", num_cls=NUM_CLASS, epsilon=EPSILON
            )
            coronal_model.model.plot_model_(
                show_shapes=True,
                show_dtype=True,
                dpi=140,
                show_layer_activations=True,
                to_file=MODEL_DIR + c_model_file,
            )
            with open(MODEL_DIR + c_model_summary, "w") as f:
                with redirect_stdout(f):
                    coronal_model.model.summary(show_trainable=True, expand_nested=True)

            coronal_model.model.summary(show_trainable=True, expand_nested=True)
            
            train_data_coronal = coronal_model.load_generator_(
                DATA_DIR, "train", batch_size=BATCH_SIZE, orientation="LAS"
            )

            validation_data_coronal = coronal_model.load_generator_(
                DATA_DIR, "val", batch_size=BATCH_SIZE, orientation="LAS"
            )

            return {
                "model": coronal_model.model,
                "train_data": train_data_coronal,
                "validation_data": validation_data_coronal,
            }

        elif plane == "axial":

            a_model_file = f"/axial_model.png"
            a_model_summary = f"/axial_model_summary.txt"

            axial_model = ResidualDenseConvNet(
                "axial", num_cls=NUM_CLASS, epsilon=EPSILON
            )
            axial_model.model.plot_model_(
                show_shapes=True,
                show_dtype=True,
                dpi=140,
                show_layer_activations=True,
                to_file=MODEL_DIR + a_model_file,
            )
            with open(MODEL_DIR + a_model_summary, "w") as f:
                with redirect_stdout(f):
                    axial_model.model.summary(show_trainable=True, expand_nested=True)

            axial_model.model.summary(show_trainable=True, expand_nested=True)

            train_data_axial = axial_model.load_generator_(
                DATA_DIR, "train", batch_size=BATCH_SIZE, orientation="LAS"
            )

            validation_data_axial = axial_model.load_generator_(
                DATA_DIR, "val", batch_size=BATCH_SIZE, orientation="LAS"
            )

            return {
                "model": axial_model.model,
                "train_data": train_data_axial,
                "validation_data": validation_data_axial,
            }
        else:
            raise ValueError("The plane must be one of 'sagittal','coronal' or 'axial'")

    def train(self, plane):
        callbacks = self.load_callbacks(plane)

        message1 = (
            f"\n\n >> Loading model and data generators for '{plane}' training......\n"
        )
        print(message1)
        # print("-" * len(message1), "\n")

        model_and_data = self.load_model_and_data(plane)
        model, train_generator, val_generator = (
            model_and_data["model"],
            model_and_data["train_data"],
            model_and_data["validation_data"],
        )
        if plane == "sagittal":

            print((("-" * 40).center(156)))
            print((("-" * 40).center(156)))
            print("Starting Sagittal training...".upper().center(156))
            print((("-" * 40).center(156)))
            print((("-" * 40).center(156)))

            with tf.device("/device:GPU:0"):
                history_sagittal = model.fit(
                    x=train_generator,
                    validation_data=val_generator,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    callbacks=[
                        callbacks["model_checkpoint"],
                        callbacks["per_epoch_predictions"],
                        callbacks["training_time"],
                        callbacks["earlystopping"],
                        callbacks["csv_logger"],
                        callbacks["leaning_rate"],
                    ],
                )

        elif plane == "coronal":

            print((("-" * 40).center(156)))
            print((("-" * 40).center(156)))
            print("Starting Coronal training...".upper().center(156))
            print((("-" * 40).center(156)))
            print((("-" * 40).center(156)))

            with tf.device("/device:GPU:0"):
                history_coronal = model.fit(
                    x=train_generator,
                    validation_data=val_generator,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    callbacks=[
                        callbacks["model_checkpoint"],
                        callbacks["per_epoch_predictions"],
                        callbacks["training_time"],
                        callbacks["earlystopping"],
                        callbacks["csv_logger"],
                        callbacks["leaning_rate"],
                    ],
                )

        elif plane == "axial":

            print((("-" * 40).center(156)))
            print((("-" * 40).center(156)))
            print("Starting Axial training...".upper().center(156))
            print((("-" * 40).center(156)))
            print((("-" * 40).center(156)))

            with tf.device("/device:GPU:0"):
                history_axial = model.fit(
                    x=train_generator,
                    validation_data=val_generator,
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    callbacks=[
                        callbacks["model_checkpoint"],
                        callbacks["per_epoch_predictions"],
                        callbacks["training_time"],
                        callbacks["earlystopping"],
                        callbacks["csv_logger"],
                        callbacks["leaning_rate"],
                    ],
                )

        else:
            raise ValueError("The plane must be one of 'sagittal','coronal' or 'axial'")
