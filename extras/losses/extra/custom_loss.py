import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.ndimage import distance_transform_edt

from .utils import calculate_one_hot_

EPSILON = 0.0001
NUM_CLASS = 5


# Calculate the false_negative and false_positive ratios w.r.t. the union and the intersection


def weighted_fn_fp_ratio(epsilon=EPSILON, num_cls=NUM_CLASS):
    def fn_fp_ratio(y_true, y_pred):

        _, _, true_positive, false_positive, false_negative, _ = calculate_one_hot_(
            y_true, y_pred
        )
        true_positive1 = tf.reduce_sum(true_positive, [1, 2], keepdims=True)

        total_false = false_negative + false_positive
        union = true_positive + total_false

        total_false = tf.reduce_sum(total_false, [1, 2], keepdims=True)
        union = tf.reduce_sum(union, [1, 2], keepdims=True)

        fn_fp_ratio1 = total_false / (union + epsilon)
        factor1 = tf.reduce_sum(fn_fp_ratio1, -1, keepdims=True)
        class_weights1 = fn_fp_ratio1 / (factor1 + epsilon)

        fn_fp_ratio2 = total_false / (true_positive1 + epsilon)
        factor2 = tf.reduce_sum(fn_fp_ratio2, -1, keepdims=True)
        class_weights2 = fn_fp_ratio2 / (factor2 + epsilon)

        fn_fp_ratio = (
            fn_fp_ratio1 * class_weights1
            + tf.math.log(fn_fp_ratio2 * class_weights2 + epsilon)
            - tf.math.log(epsilon)
        )
        fn_fp_ratio = K.sum(fn_fp_ratio)

        return fn_fp_ratio

    return fn_fp_ratio


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def weighted_combo_loss(num_cls=NUM_CLASS, one_hot=True, epsilon=EPSILON, alpha=0.5):
    def combo_loss(y_true, y_pred):

        y_true = tf.squeeze(y_true, [-1])
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_cls, dtype=tf.float32)

        y_pred_argmaxed = tf.math.argmax(y_pred, -1)
        y_pred_hot = tf.one_hot(
            tf.cast(y_pred_argmaxed, tf.int32), num_cls, dtype=tf.float32
        )

        area = tf.reduce_sum(y_true, [1, 2], keepdims=True)
        size = tf.cast(tf.size(false_negative[0]) / num_cls, tf.float32)
        proportion = area / size
        class_weights = 1 / (proportion + epsilon)

        factor = tf.reduce_sum(class_weights, axis=-1, keepdims=True)
        class_weights = class_weights / factor

        y_pred = K.clip(
            y_pred, K.epsilon(), 1 - K.epsilon()
        )  # To avoid unwanted behaviour in K.log(y_pred)

        # p * log(p̂) * class_weights
        wce_loss = y_true * tf.cast(K.log(y_pred), tf.float32)  # * class_weights
        wce_loss = -tf.divide(K.sum(wce_loss), num_cls)

        numerator = y_true * y_pred_hot  # * class_weights  # Broadcasting
        numerator = 2.0 * K.sum(numerator)

        denominator = y_true + y_pred_hot  # * class_weights # Broadcasting
        denominator = K.sum(denominator)

        dice_similarity = numerator / denominator

        return alpha * wce_loss - (1 - alpha) * dice_similarity

    return combo_loss
