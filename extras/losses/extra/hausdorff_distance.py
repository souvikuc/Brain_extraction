import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.ndimage import distance_transform_edt
from tensorflow.keras.losses import Loss
from .utils import calculate_one_hot_, normalize_class_tuple, normalize_weight_tuple
from ...config import NUM_CLASS, EPSILON


def hausdorff_distance(num_cls=NUM_CLASS, include_hard=False):
    def hausdorff(y_true, y_pred):

        _, false_positive, false_negative, _ = calculate_confusion_(
            y_true, y_pred, num_cls=NUM_CLASS
        )
        total_false = false_negative + false_positive
        total_false_brain = total_false[:, :, :, -1]
        #         print(total_false_brain)

        y_true = np.squeeze(y_true, -1)
        y_pred = np.squeeze(y_pred, -1)

        y_true_DT = np.empty_like(y_true)
        y_pred_DT = np.empty_like(y_pred)

        for i in range(y_true.shape[0]):
            y_true_DT[i] = distance_transform_edt(y_true[i])
            y_pred_DT[i] = distance_transform_edt(y_pred[i])

        #         print(y_true_DT)
        #         print(y_pred_DT)

        hausedorff_true_pred = np.max(y_true_DT * total_false_brain)
        print(hausedorff_true_pred)

        hausedorff_pred_true = np.max(y_pred_DT * total_false_brain)
        print(hausedorff_pred_true)

        return (hausedorff_true_pred + hausedorff_pred_true) / 2

    return hausdorff
