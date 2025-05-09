import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

from .utils import (
    calculate_one_hot_,
    normalize_class_tuple,
    normalize_weight_tuple,
)
from .....file_utils.config_utils import read_configs

configs = read_configs()

NUM_CLASS = configs["NUM_CLASS"]
EPSILON = configs["EPSILON"]
INPUT_SHAPE = configs["INPUT_SHAPE"]


# ------------------------------------------------------------------------------
# Probability distribution based loss functions
# ------------------------------------------------------------------------------


class Multiclass_Weighted_FocalLoss(Loss):
    """
    Multi-class weighted cross entropy.
        WCE(p, p̂) = −Σp*log(p̂)*class_weights
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param num_cls: number of classes
    :param epsilon: smoothing factor
    :return: Weighted cross entropy loss f)
    """

    #     """
    #     Computes the weighted cross entropy.
    #     :param y_true: Ground truth (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    #     :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    #     :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
    #     """

    def __init__(self, gamma=2.0, num_cls=NUM_CLASS, epsilon=EPSILON, **kwargs):
        super(Multiclass_Weighted_FocalLoss, self).__init__(**kwargs)
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.gamma = gamma

    def call(self, y_true, y_pred):

        # tranforming into one hot
        y_true_squeezed = tf.squeeze(y_true, [-1])
        y_true_hot = tf.one_hot(
            tf.cast(y_true_squeezed, tf.int32), self.num_cls, dtype=tf.float32
        )

        # class weights based on inverse of ground truth area per class (normalized)
        area = tf.reduce_sum(y_true_hot, [1, 2], keepdims=True)
        size = tf.cast(tf.size(y_true[0]), tf.float32)
        proportion = area / size
        class_weights = tf.cast(1 / (proportion + self.epsilon), tf.float32)
        factor = tf.reduce_sum(class_weights, axis=-1, keepdims=True)
        class_weights = class_weights / factor

        # To avoid unwanted behaviour in K.log(y_pred)
        y_pred = K.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # p * log(p̂) * class_weights
        wce_loss = (
            y_true_hot
            * tf.cast(K.log(y_pred), tf.float32)
            * (class_weights + (1 - y_pred) ** self.gamma)
        )
        wce_loss = tf.divide(K.sum(wce_loss), self.num_cls)

        return tf.math.log(-wce_loss)  # scaling with logarithm


class Pixelwise_Weighted_FocalLoss(Loss):
    """
    Multi-class weighted cross entropy.
        WCE(p, p̂) = −Σp*log(p̂)*class_weights
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param num_cls: number of classes
    :param epsilon: smoothing factor
    :return: Weighted cross entropy loss f)
    """

    #     """
    #     Computes the weighted cross entropy.
    #     :param y_true: Ground truth (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    #     :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    #     :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
    #     """

    def __init__(
        self, weights, gamma=2.0, num_cls=NUM_CLASS, epsilon=EPSILON, **kwargs
    ):
        super(Pixelwise_Weighted_FocalLoss, self).__init__(**kwargs)
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.wts = weights
        self.gamma = gamma

    def call(self, y_true, y_pred):

        # tranforming into one hot
        y_true_squeezed = tf.squeeze(y_true, [-1])
        y_true_hot = tf.one_hot(
            tf.cast(y_true_squeezed, tf.int32), self.num_cls, dtype=tf.float32
        )

        # To avoid unwanted behaviour in K.log(y_pred)
        y_pred = K.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # p * log(p̂) * class_weights
        wce_loss = (
            y_true_hot
            * tf.cast(K.log(y_pred), tf.float32)
            * (self.wts + (1.0 - y_pred) ** self.gamma)
        )
        wce_loss = tf.divide(K.sum(wce_loss), self.num_cls)

        return tf.math.log(-wce_loss)  # scaling with logarithm
        # return -wce_loss


class Multiclass_Weighted_CE(Loss):
    """
    Multi-class weighted cross entropy.
        WCE(p, p̂) = −Σp*log(p̂)*class_weights
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param num_cls: number of classes
    :param epsilon: smoothing factor
    :return: Weighted cross entropy loss f)
    """

    #     """
    #     Computes the weighted cross entropy.
    #     :param y_true: Ground truth (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    #     :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    #     :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
    #     """

    def __init__(self, num_cls=NUM_CLASS, epsilon=EPSILON, **kwargs):
        super(Multiclass_Weighted_CE, self).__init__(**kwargs)
        self.num_cls = num_cls
        self.epsilon = epsilon

    def call(self, y_true, y_pred):

        # tranforming into one hot
        y_true_squeezed = tf.squeeze(y_true, [-1])
        y_true_hot = tf.one_hot(
            tf.cast(y_true_squeezed, tf.int32), self.num_cls, dtype=tf.float32
        )

        # class weights based on inverse of ground truth area per class (normalized)
        area = tf.reduce_sum(y_true_hot, [1, 2], keepdims=True)
        size = tf.cast(tf.size(y_true[0]), tf.float32)
        proportion = area / size
        class_weights = tf.cast(1 / (proportion + self.epsilon), tf.float32)
        factor = tf.reduce_sum(class_weights, axis=-1, keepdims=True)
        class_weights = class_weights / factor

        # To avoid unwanted behaviour in K.log(y_pred)
        y_pred = K.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # p * log(p̂) * class_weights
        wce_loss = y_true_hot * tf.cast(K.log(y_pred), tf.float32) * class_weights
        wce_loss = tf.divide(K.sum(wce_loss), self.num_cls)

        return tf.math.log(-wce_loss)  # scaling with logarithm


class Pixelwise_Weighted_CE(Loss):
    """
    Multi-class weighted cross entropy.
        WCE(p, p̂) = −Σp*log(p̂)*class_weights
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param num_cls: number of classes
    :param epsilon: smoothing factor
    :return: Weighted cross entropy loss f)
    """

    #     """
    #     Computes the weighted cross entropy.
    #     :param y_true: Ground truth (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CHANNELS>))
    #     :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    #     :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
    #     """

    def __init__(self, weights, num_cls=NUM_CLASS, epsilon=EPSILON, **kwargs):
        super(Pixelwise_Weighted_CE, self).__init__(**kwargs)
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.wts = weights

    def call(self, y_true, y_pred):

        # tranforming into one hot
        y_true_squeezed = tf.squeeze(y_true, [-1])
        y_true_hot = tf.one_hot(
            tf.cast(y_true_squeezed, tf.int32), self.num_cls, dtype=tf.float32
        )

        # To avoid unwanted behaviour in K.log(y_pred)
        y_pred = K.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # p * log(p̂) * class_weights
        wce_loss = y_true_hot * tf.cast(K.log(y_pred), tf.float32) * (self.wts + 1.0)
        wce_loss = tf.divide(K.sum(wce_loss), self.num_cls)

        return tf.math.log(-wce_loss)  # scaling with logarithm
        # return -wce_loss


# ------------------------------------------------------------------------------
# Regions overlap-based loss functions
# ------------------------------------------------------------------------------


class Multi_Weighted_SoftMax_DiceLoss(Loss):
    def __init__(
        self,
        num_cls=NUM_CLASS,
        epsilon=EPSILON,
        include_class=None,
        class_weights=None,
        **kwargs
    ):
        super(Multi_Weighted_SoftMax_DiceLoss, self).__init__(**kwargs)
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.include_class = normalize_class_tuple(
            include_class, self.num_cls, "include_class"
        )
        self.class_weights = normalize_weight_tuple(
            class_weights,
            len(self.include_class),
            "class_weights",
            self.include_class,
        )

    def call(self, y_true, y_pred):

        y_true_hot = tf.one_hot(tf.cast(tf.squeeze(y_true, -1), tf.int32), self.num_cls)

        if self.class_weights != None:
            cls_wts = K.cast(K.reshape(self.class_weights, (1, 1, 1, -1)), tf.float32)
        else:
            gt_area_per_class = K.sum(
                y_true_hot,
                [1, 2],
                keepdims=True,
            )
            size = K.cast(tf.size(y_true[0]) / self.num_cls, tf.float32)
            area_prop_per_class = gt_area_per_class / size
            inv_gt_area_prop_per_class = 1 / (area_prop_per_class + self.epsilon)
            factor = K.sum(inv_gt_area_prop_per_class, axis=-1, keepdims=True)
            cls_wts = inv_gt_area_prop_per_class / factor
            cls_wts = tf.gather(cls_wts, self.include_class, axis=-1)

        selected_y_true_hot = tf.gather(y_true_hot, self.include_class, axis=-1)
        selected_y_pred_hot = tf.gather(y_pred, self.include_class, axis=-1)

        numerator = 2.0 * K.sum(
            selected_y_true_hot * selected_y_pred_hot,
            [1, 2],
            keepdims=True,
        )  # Broadcasting
        denominator = K.sum(
            selected_y_true_hot + selected_y_pred_hot,
            [1, 2],
            keepdims=True,
        )  # Broadcasting

        d_similarity = numerator / (denominator + self.epsilon)
        dice_similarity = K.mean(d_similarity * cls_wts)
        return 1 - dice_similarity
