import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, Input, Add, Concatenate
from tensorflow.keras import Model

from .utils import normalize_merge


class GlobalConv2D(Layer):

    """inspired by https://arxiv.org/pdf/1703.02719.pdf"""

    def __init__(self, kernels=256, kernel_param=5, merge="sum", **kwargs):
        super(GlobalConv2D, self).__init__(**kwargs)
        self.kernels = kernels
        self.kernels_param = kernel_param
        self.merge = normalize_merge(merge)
        self.conv11 = Conv2D(
            self.kernels,
            kernel_size=(self.kernels_param, 1),
            padding="valid",
            activation="relu",
        )
        self.conv12 = Conv2D(
            self.kernels,
            kernel_size=(1, self.kernels_param),
            padding="valid",
            activation="relu",
        )
        self.conv21 = Conv2D(
            self.kernels,
            kernel_size=(1, self.kernels_param),
            padding="valid",
            activation="relu",
        )
        self.conv22 = Conv2D(
            self.kernels,
            kernel_size=(self.kernels_param, 1),
            padding="valid",
            activation="relu",
        )
        if self.merge == "sum":
            self.add = Add()
        else:
            self.concat = Concatenate()

    def call(self, inputs):
        x1 = self.conv11(inputs)
        x2 = self.conv21(inputs)

        x1 = self.conv12(x1)
        x2 = self.conv22(x2)

        if self.merge == "sum":
            glob_conv_output = self.add([x1, x2])
        else:
            glob_conv_output = self.concat([x1, x2])

        return glob_conv_output

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(
            inputs=inputs, outputs=self.call(inputs), name="global_convolution_layer"
        )
        return m.summary(**kwargs)
