import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    Add,
    Resizing,
)
from ...config import INPUT_SHAPE
from tensorflow.keras import Model
from .utils import normalize_tuple, normalize_data_format
from tensorflow.keras.layers import Input


class AtrousConvPyramid2D(Layer):
    def __init__(
        self,
        dilation_rates=[2, 3, 6],
        filters=256,
        kernel_size=3,
        strides=(1, 1),
        data_format=None,
        activation="relu",
        upscale="bilinear",
        out_channel = None,
        **kwargs,
    ):

        super(AtrousConvPyramid2D, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.upscale = upscale
        self.a_conv_outputs = None
        self.activation = activation
        self.kernel_size = normalize_tuple(kernel_size, 2, "kernel")
        self.data_format = normalize_data_format(data_format)
        self.out_channel= out_channel
        self.dilation_rates = [
            normalize_tuple(dilation_rate, 2, "dilation_rates")
            for dilation_rate in dilation_rates
        ]
        self.a_conv_layers = [
            Conv2D(
                self.filters,
                self.kernel_size,
                self.strides,
                padding="valid",
                data_format=self.data_format,
                activation=self.activation,
                dilation_rate=self.dilation_rates[i],
            )
            for i in range(len(self.dilation_rates))
        ]
        self.conv1 = [
            Conv2D(1, 1, activation=self.activation)
            for _ in range(len(self.dilation_rates))
        ]
        self.conv2 = Conv2D(1, 1, activation=self.activation)
        self.conv3 = None
        self.resize = None

    # def build(self, input_shape):

    # self.H, self.W, self.C = input_shape[1:]

    # self.resize = [
    #     Resizing(self.H, self.W, interpolation=self.upscale)
    #     for _ in range(len(self.dilation_rates))
    # ]
    # self.conv3 = Conv2D(
    #     self.C, 1, activation=self.activation, name="astrous_output"
    # )

    def call(self, inputs):
        self.H, self.W, self.C = inputs.shape[1:]
        if self.resize == None:
            self.resize = [
                Resizing(self.H, self.W, interpolation=self.upscale)
                for _ in range(len(self.dilation_rates))
            ]
        if self.conv3 == None:
            if self.out_channel == None:
                self.conv3 = Conv2D(
                    self.C, 1, activation=self.activation, name="astrous_output"
                )
            else:
                self.conv3 = Conv2D(
                    self.out_channel, 1, activation=self.activation, name="astrous_output"
                )

        if self.a_conv_outputs == None:
            self.a_conv_outputs = []

        self.a_conv_outputs = [
            self.a_conv_layers[i](inputs) for i in range(len(self.dilation_rates))
        ]

        self.a_conv_outputs = [
            self.conv1[i](self.a_conv_outputs[i])
            for i in range(len(self.a_conv_outputs))
        ]
        self.a_conv_outputs = [
            self.resize[i](self.a_conv_outputs[i])
            for i in range(len(self.a_conv_outputs))
        ]

        concated_a_conv_outputs = tf.concat(self.a_conv_outputs, -1)
        input1 = self.conv2(inputs)
        final_concated = tf.concat([concated_a_conv_outputs, input1], -1)
        result = self.conv3(final_concated)

        return result

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(
            inputs=inputs, outputs=self.call(inputs), name="atrous_convolution_model"
        )
        return m.summary(**kwargs)
