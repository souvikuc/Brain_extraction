import tensorflow as tf
import pathlib, os, glob, h5py
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric

from .utils import (
    calculate_one_hot_,
    normalize_class_tuple,
    normalize_weight_tuple,
)
from .....file_utils.config_utils import read_configs

configs = read_configs()
NUM_CLASS = configs["NUM_CLASS"]
EPSILON = configs["EPSILON"]


class Multiclass_Weighted_SoftMax_DiceSimilarity(Metric):

    """
    Compute weighted Dice loss.
    :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    :return: Weighted squared Dice loss (tf.Tensor, shape=(None,))
    """

    def __init__(
        self,
        num_cls=NUM_CLASS,
        epsilon=EPSILON,
        include_class=None,
        class_weights=None,
        name="multi_weighted_dice_similarity",
        **kwargs
    ):
        super(Multiclass_Weighted_SoftMax_DiceSimilarity, self).__init__(
            name=name, **kwargs
        )
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.include_class = normalize_class_tuple(
            include_class, self.num_cls, "include_class"
        )
        self.class_weights = normalize_weight_tuple(
            class_weights, len(self.include_class), "weights", "include_class"
        )
        self.d_sim = self.add_weight(
            name="weighted_dice_similarity", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):

        # transforming into one hot representation
        y_true_hot, y_pred_hot = calculate_one_hot_(
            y_true, y_pred, self.num_cls, confusion_per_class=False
        )

        # filtering the values for given classes
        selected_y_true_hot = tf.gather(y_true_hot, self.include_class, axis=-1)
        selected_y_pred_hot = tf.gather(y_pred_hot, self.include_class, axis=-1)

        # numerator of the dice loss
        numerator = 2.0 * tf.reduce_sum(
            selected_y_true_hot * selected_y_pred_hot, [1, 2], keepdims=True
        )  # Broadcasting

        # denominator of dice loss
        denominator = tf.reduce_sum(
            selected_y_true_hot + selected_y_pred_hot, [1, 2], keepdims=True
        )  # Broadcasting

        # un-weighted_dice_similarity
        d_similarity = numerator / (denominator + self.epsilon)

        if self.class_weights == None:

            # calculating the weights based on ground truth area each class covers (normalized)
            gt_area_per_class = K.sum(selected_y_true_hot, [1, 2], keepdims=True)
            size = tf.cast(tf.size(y_true[0, :, :, 0]), tf.float32)
            area_prop_per_class = gt_area_per_class / size
            inv_gt_area_prop_per_class = 1 / (area_prop_per_class + self.epsilon)
            factor = K.sum(inv_gt_area_prop_per_class, axis=-1, keepdims=True)
            wts = inv_gt_area_prop_per_class / factor

            # weighted dice similarity
            dice_similarity = K.mean(d_similarity * wts)
            self.d_sim.assign(dice_similarity)

        else:

            # calculating weighted dice similarity from user fed weights
            wts = tf.cast(tf.reshape(self.class_weights, (1, 1, 1, -1)), tf.float32)
            dice_similarity = K.mean(d_similarity * wts)
            self.d_sim.assign(dice_similarity)

    def result(self):
        return self.d_sim

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.d_sim.assign(0.0)


class Batchwise_Avg_DiceSimilarity(Metric):

    """
    Compute weighted Dice loss.
    :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    :return: Weighted squared Dice loss (tf.Tensor, shape=(None,))
    """

    def __init__(
        self,
        num_cls=NUM_CLASS,
        epsilon=EPSILON,
        name="batchwise_avg_dice_similarity",
        **kwargs
    ):
        super(Batchwise_Avg_DiceSimilarity, self).__init__(name=name, **kwargs)
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.d_sim = self.add_weight(
            name="weighted_dice_similarity", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true_hot, y_pred_hot = calculate_one_hot_(
            y_true, y_pred, self.num_cls, confusion_per_class=False
        )
        # numerator of the dice loss
        numerator = 2.0 * tf.reduce_sum(y_true_hot * y_pred_hot, [1, 2, 3])
        # denominator of dice loss
        denominator = tf.reduce_sum(y_true_hot + y_pred_hot, [1, 2, 3])  # Broadcasting

        d_similarity = numerator / (denominator + self.epsilon)
        dice_similarity = K.mean(d_similarity)
        self.d_sim.assign(dice_similarity)

    def result(self):
        return self.d_sim

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.d_sim.assign(0.0)
