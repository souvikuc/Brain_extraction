import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Layer,
    Input,
    Resizing,
    MaxPooling2D,
    AveragePooling2D,
)
from tensorflow.keras import Model

from tensorflow.keras import backend as K

from .layers import MixedPooling2D
from .....file_utils.config_utils import read_configs

configs = read_configs()

EPSILON = configs["EPSILON"]
NUM_CLASS = configs["NUM_CLASS"]


def _normalize(values):
    values_np = np.array(values)
    tot = np.sum(values_np)
    result = values_np / tot
    return tuple(result)


def normalize_weight_tuple(value, included_num_clss, name1, name2):

    if isinstance(value, (int, float)):
        value_tuple = (value,) * included_num_clss

    elif value == None:
        return value

    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                f"{name1} must be a tuple of weights of same length as included class:{name2} "
            )

        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"The {name1} must be a tuple of weights(int or float). Will be normalized either way. "
                )
        if len(value_tuple) == 1:
            value_tuple = (value_tuple[0],) * included_num_clss
        elif len(value_tuple) != included_num_clss:
            raise ValueError(
                f"The length of {name1} must be same as included class:{name2}"
            )

    return _normalize(value_tuple)


def calculate_one_hot_(y_true, y_pred, num_cls=NUM_CLASS, confusion_per_class=True):

    y_true = K.squeeze(y_true, -1)
    y_true = K.one_hot(K.cast(y_true, tf.int32), num_cls)
    y_pred = K.squeeze(y_pred, -1)
    y_pred = K.one_hot(K.cast(y_pred, tf.int32), num_cls)

    if confusion_per_class:
        true_positive = y_true * y_pred
        false_positive = y_pred - true_positive
        false_negative = y_true - true_positive
        true_negative = K.cast((y_true + y_pred) == 0, tf.float32)
        return (
            y_true,
            y_pred,
            true_positive,
            false_positive,
            false_negative,
            true_negative,
        )
    return y_true, y_pred


def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.
    A copy of tensorflow.python.keras.util.
    Args:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
    Returns:
      A tuple of n integers.
    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple


def normalize_class_tuple(value, tot_num_class, name):

    if value == None:
        return tuple(range(tot_num_class))

    if isinstance(value, int):
        if value in range(tot_num_class):
            return (value,)  #### * tot_num_class
        else:
            raise ValueError(
                f"{name} must be single integer or tuple of integers specifying the class labels"
            )
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError(f"The {name} must be a tuple of class indices(integers)")

        if len(set(value_tuple)) < len(value_tuple):
            raise ValueError(f"There must be no duplicate class indices in {name}")

        if len(value_tuple) > tot_num_class:
            raise ValueError(
                f"The length of {name} can't exceed total number of classes"
            )

        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"The {name} must be a tuple of class indices(integers)"
                )
            if single_value not in range(tot_num_class):
                raise ValueError(f"{single_value} is not a valid class label ")

            if single_value >= tot_num_class:
                raise ValueError(f"{single_value} is out of bounds of class labels ")

        return value_tuple


def normalize_data_format(value):
    if value is None:
        value = tf.keras.backend.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            '"channels_first", "channels_last". Received: ' + str(value)
        )
    return data_format


def normalize_merge(value):
    if value is None:
        value = "max"
    merge = value.lower()
    if merge not in {"max", "sum", "concat"}:
        raise ValueError(
            "The `merge` argument must be one of "
            '"max", "sum", "concat". Received: ' + str(value)
        )
    return merge


def normalize_pooling(value):
    if value is None:
        value = "max"
    pooling = value.lower()
    if pooling not in {"max", "mixed", "spp"}:
        raise ValueError(
            "The `pooling` argument must be one of "
            '"max", "mixed", "spp". Received: ' + str(value)
        )
    return pooling


def normalize_unpooling(value):
    if value is None:
        value = "deconv"
    unpooling = value.lower()
    if unpooling not in {"deconv", "upsample"}:
        raise ValueError(
            "The `unpooling` argument must be one of "
            '"deconv", "upsample". Received: ' + str(value)
        )
    return unpooling


def normalize_padding(value):
    if value is None:
        value = "valid"
    pad = value.lower()
    if pad not in {"valid", "same"}:
        raise ValueError(
            "The `padding` argument must be one of "
            '"valid", "same". Received: ' + str(value)
        )
    return pad


def normalize_pool_type(value):
    if value is None:
        value = "max"
    pool_type = value.lower()
    if pool_type not in {"max", "mean", "avg", "mixed"}:
        raise ValueError(
            "The `pool_type` argument must be one of "
            '"max", "mean", "avg", "mixed". Received: ' + str(value)
        )
    return pool_type


def validate_input_shapes(values):
    if len(set(values)) != 1:
        raise ValueError("All the input_shapes must be same")


class Pooling_Op(Layer):
    def __init__(
        self,
        pool_size=2,
        strides=2,
        pool_type="max",
        data_format=None,
        padding="valid",
        out_size=None,
        upscale="bilinear",
        alpha=0.5,
        **kwargs,
    ):
        super(Pooling_Op, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.pool_type = normalize_pool_type(pool_type)
        self.data_format = normalize_data_format(data_format)
        self.padding = padding
        self.out_size = out_size
        self.upscale = upscale
        self.alpha = alpha

        if self.pool_type == "max":
            self.pooled = MaxPooling2D(
                self.pool_size,
                self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        elif (self.pool_type == "avg") | (self.pool_type == "mean"):
            self.pooled = AveragePooling2D(
                self.pool_size,
                self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        else:
            self.pooled = MixedPooling2D(
                self.pool_size,
                self.strides,
                data_format=self.data_format,
                padding=self.padding,
                alpha=self.alpha,
            )
        if self.out_size is not None:
            self.resize = Resizing(*out_size, interpolation=self.upscale)

    def call(self, inputs):
        pooled = self.pooled(inputs)

        if self.out_size is not None:
            output = self.resize(pooled)
            return output
        else:
            return pooled

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs))
        return m.summary(**kwargs)
