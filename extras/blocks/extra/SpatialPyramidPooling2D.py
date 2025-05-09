import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, Input, Add, Concatenate
from tensorflow.keras import Model

from .utils import (
    normalize_tuple,
    normalize_data_format,
    normalize_merge,
    normalize_padding,
)
from .utils import Pooling_Op


class SpatialPyramidPooling2D(Layer):
    def __init__(
        self,
        bins,
        data_format=None,
        pool_type="max",
        alpha=0.5,
        merge="sum",
        padding="valid",
        weight=None,
        output_shape=None,
        upscale="bilinear",
        **kwargs,
    ):

        super(SpatialPyramidPooling2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.pool_type = pool_type
        self.merge = normalize_merge(merge)
        self.conv = None
        self.pool_op = None
        self.padding = normalize_padding(padding)
        self.out_shape = output_shape
        self.upscale = upscale
        self.bins = [normalize_tuple(bin, 2, "bin") for bin in bins]
        self.data_format = normalize_data_format(data_format)

        if self.merge == "sum":
            self.add = Add()
        else:
            self.concat = Concatenate()

        if weight == None:
            self.weight = tf.ones((len(self.bins), 1, 1, 1, 1))
        else:
            self.wght = tf.convert_to_tensor(weight, dtype=tf.float32)
            self.weight = tf.reshape(
                self.wght,
                (self.wght.shape[0], 1, 1, 1, 1),
            )

    def build(self, input_shape):
        tf_bins = tf.convert_to_tensor(self.bins)
        shape = tf.reshape(input_shape[1:-1], (1, 2))
        res = tf.cast(shape // tf_bins, tf.bool)

        if not tf.reduce_all(res):
            raise ValueError("The bin_size is greater than the input size")

    def _calculate_output_shape(
        self,
        height,
        width,
        pool_size=2,
        stride=2,
        padding=None,
    ):
        if padding is None:
            padding = self.padding

        if self.padding == "valid":
            H = ((height - pool_size) // stride) + 1
            W = ((width - pool_size) // stride) + 1
        else:
            H = ((height - 1) // stride) + 1
            W = ((width - 1) // stride) + 1
        return H, W

    def call(self, inputs):

        self.H, self.W, self.C = inputs.shape[1:]

        if self.out_shape == None:
            outp_shape = self._calculate_output_shape(
                self.H, self.W, pool_size=2, stride=2, padding=self.padding
            )
        else:
            outp_shape = normalize_tuple(self.out_shape, 2, "output_shape")

        self.pool_size = [
            (math.ceil(self.H / bin[0]), math.ceil(self.W / bin[1]))
            for bin in self.bins
        ]
        self.strides = [(self.H // bin[0], self.W // bin[1]) for bin in self.bins]
        if self.pool_op is None:
            self.pool_op = [
                Pooling_Op(
                    pool_type=self.pool_type,
                    pool_size=pool_size,
                    strides=strides,
                    data_format=self.data_format,
                    padding=self.padding,
                    out_size=outp_shape,
                    upscale=self.upscale,
                    alpha=self.alpha,
                )
                for pool_size, strides in zip(self.pool_size, self.strides)
            ]

        pool_outputs = [self.pool_op[i](inputs) for i in range(len(self.pool_op))]

        if self.merge == "sum":
            assert self.weight.shape[0] == len(
                self.bins
            ), "weights must be 1-D array of same length as the number of pooling operations, first dimension of bins"
            final_outputs = self.add(pool_outputs)
            final_outputs = final_outputs * self.weight
            final_outputs = tf.reduce_sum(final_outputs, 0)
        else:
            final_outputs = self.concat(pool_outputs, -1)

        if self.conv is None:
            self.conv = Conv2D(
                self.C, 1, kernel_initializer="glorot_uniform", name="spp_output"
            )
        result = self.conv(final_outputs)

        return result

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="spp_model")
        return m.summary(**kwargs)
