import math
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    ReLU,
    Layer,
    Input,
    PReLU,
    Conv2D,
    Lambda,
    Reshape,
    Maximum,
    Resizing,
    Multiply,
    Activation,
    Concatenate,
    UpSampling2D,
    Conv2DTranspose,
    BatchNormalization,
    GlobalAveragePooling2D,
)

from .....file_utils.config_utils import read_configs
from .utils import (
    Pooling_Op,
    normalize_tuple,
    normalize_merge,
    normalize_padding,
    normalize_pool_type,
    normalize_data_format,
)

configs = read_configs()
INPUT_SHAPE = configs["INPUT_SHAPE"]


class Merge(Layer):
    def __init__(self, kind="sum", activation="relu", out_channel=256, **kwargs):

        super(Merge, self).__init__(**kwargs)
        self.kind = normalize_merge(kind)
        self.activation = activation
        self.out_channel = out_channel

        if self.kind == "max":
            self.merge = Maximum(name="merged_max")
        elif self.kind == "concat":
            self.merge = Concatenate(name="merged_concat")
        else:
            self.merge = Add(name="merged_add")

        self.conv1 = Conv2D(
            self.out_channel,
            3,
            padding="same",
            activation=self.activation,
            dtype=tf.float32,
            name="merged_conv",
        )

    def call(self, inputs):
        merged = self.merge(inputs)
        conv1 = self.conv1(merged)

        return conv1


class GlobalConv2D(Layer):

    """inspired by https://arxiv.org/pdf/1703.02719.pdf"""

    def __init__(
        self, kernels=256, kernel_param=5, merge="sum", padding="valid", **kwargs
    ):
        super(GlobalConv2D, self).__init__(**kwargs)
        self.kernels = kernels
        self.padding = normalize_padding(padding)
        self.kernels_param = kernel_param
        self.merge = normalize_merge(merge)
        self.conv11 = Conv2D(
            self.kernels,
            kernel_size=(self.kernels_param, 1),
            padding=self.padding,
            activation="relu",
        )
        self.bn11 = BatchNormalization()
        self.conv12 = Conv2D(
            self.kernels,
            kernel_size=(1, self.kernels_param),
            padding=self.padding,
            activation="relu",
        )
        self.bn12 = BatchNormalization()

        self.conv21 = Conv2D(
            self.kernels,
            kernel_size=(1, self.kernels_param),
            padding=self.padding,
            activation="relu",
        )
        self.bn21 = BatchNormalization()
        self.conv22 = Conv2D(
            self.kernels,
            kernel_size=(self.kernels_param, 1),
            padding=self.padding,
            activation="relu",
        )
        self.bn22 = BatchNormalization()

        # if self.merge == "sum":
        #     self.add = Add(name="global_conv_output")
        # else:
        #     self.concat = Concatenate(name="global_conv_output")

    def call(self, inputs):
        conv11 = self.conv11(inputs)
        conv21 = self.conv21(inputs)

        bn11 = self.bn11(conv11)
        bn21 = self.bn21(conv21)

        conv12 = self.conv12(bn11)
        conv22 = self.conv22(bn21)

        bn12 = self.bn11(conv12)
        bn22 = self.bn21(conv22)

        if self.merge == "sum":
            glob_conv_output = self.add([bn12, bn22])
        else:
            glob_conv_output = self.concat([bn12, bn22])

        return glob_conv_output

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(
            inputs=inputs, outputs=self.call(inputs), name="global_convolution_layer"
        )
        return m.summary(**kwargs)


class AtrousConvPyramid2D(Layer):
    """inspired by https://arxiv.org/pdf/1706.05587v3.pdf"""

    def __init__(
        self,
        dilation_rates=[1, 2, 4, 6, 8],
        filters=128,
        kernel_size=3,
        strides=(1, 1),
        data_format=None,
        activation=None,
        upscale="bicubic",
        out_channel=None,
        **kwargs,
    ):

        super(AtrousConvPyramid2D, self).__init__(**kwargs)
        self.conv3 = None
        self.resize = None
        self.filters = filters
        self.strides = strides
        self.upscale = upscale
        self.a_conv_outputs = None
        self.activation = activation
        self.out_channel = out_channel
        self.data_format = normalize_data_format(data_format)
        self.kernel_size = normalize_tuple(kernel_size, 2, "kernel")
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
        # self.bns = [BatchNormalization() for i in range(len(self.dilation_rates))]
        self.conv1 = Conv2D(256, 1, activation=self.activation)
        self.conv2 = Conv2D(256, 1, activation="sigmoid")

    def call(self, inputs):
        self.H, self.W, self.C = inputs.shape[1:]
        input1 = self.conv2(inputs)
        if self.resize == None:
            self.resize = [
                Resizing(self.H, self.W, interpolation=self.upscale)
                for _ in range(len(self.dilation_rates))
            ]
        if self.conv3 == None:
            if self.out_channel == None:
                self.conv3 = Conv2D(
                    self.C,
                    1,
                    activation=self.activation,
                    name="atrous_conv_pyramid_output",
                )
            else:
                self.conv3 = Conv2D(
                    self.out_channel,
                    1,
                    activation=self.activation,
                    name="atrous_conv_pyramid_output",
                )

        if self.a_conv_outputs == None:
            self.a_conv_outputs = []

        self.a_conv_outputs = [
            self.a_conv_layers[i](inputs) for i in range(len(self.dilation_rates))
        ]
        # self.a_conv_outputs = [
        #     self.bns[i](self.a_conv_outputs[i]) for i in range(len(self.dilation_rates))
        # ]
        self.a_conv_outputs = [
            self.resize[i](self.a_conv_outputs[i])
            for i in range(len(self.a_conv_outputs))
        ]
        concated_a_conv_outputs = tf.concat(self.a_conv_outputs, -1)
        self.a_conv_outputs = self.conv1(concated_a_conv_outputs)
        final_merged = self.a_conv_outputs * input1
        result = self.conv3(final_merged)
        return result

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(
            inputs=inputs, outputs=self.call(inputs), name="atrous_convolution_layer"
        )
        return m.summary(**kwargs)


class SpatialPyramidPooling2D(Layer):
    """inspired by https://arxiv.org/pdf/1406.4729v4.pdf"""

    def __init__(
        self,
        bins,
        data_format=None,
        pool_type="max",
        alpha=0.5,
        merge="sum",
        padding="valid",
        weight=None,
        trainable=False,
        output_shape=None,
        upscale="bilinear",
        **kwargs,
    ):

        super(SpatialPyramidPooling2D, self).__init__(**kwargs)
        self.conv = None
        self.alpha = alpha
        self.pool_op = None
        self.weight = weight
        self.upscale = upscale
        self.trainable = trainable
        self.out_shape = output_shape
        self.merge = normalize_merge(merge)
        self.padding = normalize_padding(padding)
        self.pool_type = normalize_pool_type(pool_type)
        self.data_format = normalize_data_format(data_format)
        self.bins = [normalize_tuple(bin, 2, "bin") for bin in bins]

        if self.merge == "sum":
            self.add = Add()
        elif self.merge == "max":
            self.max = Maximum()
        else:
            self.concat = Concatenate(axis=-1)

        # self.merged = Merge(kind=self.merge, activation="relu", out_channel=256)

        if self.weight == None:
            self.weight = self.add_weight(
                shape=(len(self.bins), 1, 1, 1, 1),
                initializer="ones",
                trainable=self.trainable,
            )
        else:
            self.wght1 = tf.convert_to_tensor(self.weight, dtype=tf.float32)
            self.wght2 = tf.reshape(self.wght1, (self.wght1.shape[0], 1, 1, 1, 1))
            self.weight = tf.Variable(self.wght2, trainable=self.trainable)

    def build(self, input_shape):
        tf_bins = tf.convert_to_tensor(self.bins)
        shape = tf.reshape(input_shape[1:-1], (1, 2))
        res = tf.cast(shape // tf_bins, tf.bool)

        if not tf.reduce_all(res):
            raise ValueError("The bin_size is greater than the input size")

    def calculate_output_shape_(
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
            outp_shape = self.calculate_output_shape_(
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
            ), "weights must be 1-D array of same length as the number of pooling operations,\
                first dimension of bins"
            final_outputs = self.add(pool_outputs)
            final_outputs = final_outputs * self.weight
            final_outputs = tf.reduce_sum(final_outputs, 0)
        elif self.merge == "max":
            assert self.weight.shape[0] == len(
                self.bins
            ), "weights must be 1-D array of same length as the number of pooling operations,\
                first dimension of bins"
            final_outputs = self.max(pool_outputs)
            final_outputs = final_outputs * self.weight
            final_outputs = tf.reduce_sum(final_outputs, 0)
        else:
            final_outputs = self.concat(pool_outputs)
            kernel = tf.ones((1, 1, final_outputs.shape[-1], self.C))
            final_outputs = tf.nn.conv2d(
                final_outputs, kernel, strides=[1, 1], padding="SAME"
            )

        maximums = tf.reduce_max(final_outputs, axis=[1, 2], keepdims=True)
        final_outputs = tf.where(final_outputs != 0.0, final_outputs / maximums, 0.0)

        return final_outputs

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="spp_layer")
        return m.summary(**kwargs)


class Res_Convolution_Block1(Layer):
    """inspired by https://arxiv.org/pdf/2112.09654.pdf"""

    def __init__(
        self,
        kernels=16,
        size=3,
        activation="relu",
        level="",
        idx="",
        atrous=False,
        res_merge="max",
        **kwargs,
    ):
        super(Res_Convolution_Block1, self).__init__(**kwargs)

        self.atrous = atrous
        self.res_merge = normalize_merge(res_merge)

        self.prelu0 = PReLU(name="prelu0_" + str(level) + "_" + str(idx))
        self.prelu1 = PReLU(name="prelu1_" + str(level) + "_" + str(idx))
        self.prelu2 = PReLU(name="prelu2_" + str(level) + "_" + str(idx))

        # if self.res_merge == "max":
        #     self.merge0 = Maximum(name="max0_" + str(level) + "_" + str(idx))
        #     self.merge1 = Maximum(name="max1_" + str(level) + "_" + str(idx))
        # elif self.res_merge == "concat":
        #     self.merge0 = Concatenate(name="concat0_" + str(level) + "_" + str(idx))
        #     self.merge1 = Concatenate(name="concat1_" + str(level) + "_" + str(idx))
        # else:
        #     self.merge0 = Add(name="add0_" + str(level) + "_" + str(idx))
        #     self.merge1 = Add(name="add1_" + str(level) + "_" + str(idx))

        # self.merge2 = Add(
        #     name="res_conv_block1_" + str(level) + "_" + str(idx) + "output"
        # )
        self.merge0 = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="resconv1_merge0" + str(level) + "_" + str(idx),
        )
        self.merge1 = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="resconv1_merge1" + str(level) + "_" + str(idx),
        )
        self.merge2 = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="resconv1_merge2" + str(level) + "_" + str(idx),
        )

        self.bn0 = BatchNormalization(name="bn0_" + str(level) + "_" + str(idx))
        self.bn1 = BatchNormalization(name="bn1_" + str(level) + "_" + str(idx))
        self.bn2 = BatchNormalization(name="bn2_" + str(level) + "_" + str(idx))
        self.bn3 = BatchNormalization(name="bn3_" + str(level) + "_" + str(idx))
        self.bn4 = BatchNormalization(
            name="bn4_" + str(level) + "_" + str(idx) + "_output"
        )

        self.conv0 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv0_" + str(level) + "_" + str(idx),
        )
        self.conv1 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv1_" + str(level) + "_" + str(idx),
        )
        self.conv2 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv2_" + str(level) + "_" + str(idx),
        )
        if self.atrous:
            self.aconv3 = AtrousConvPyramid2D(
                # dilation_rates=[2, 4, 6],
                dtype=tf.float32,
                name="aconv3_" + str(level) + "_" + str(idx),
            )
        else:
            self.conv3 = Conv2D(
                kernels,
                size,
                padding="same",
                activation=activation,
                dtype=tf.float32,
                name="conv3_" + str(level) + "_" + str(idx),
            )
        self.conv4 = Conv2D(
            kernels,
            1,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv4_" + str(level) + "_" + str(idx),
        )

    def call(self, inputs):

        bn0 = self.bn0(inputs)
        conv0 = self.conv0(bn0)
        bn1 = self.bn1(conv0)

        prelu0 = self.prelu0(bn1)
        conv1 = self.conv1(prelu0)
        bn2 = self.bn2(conv1)

        merged0 = self.merge0([bn2, bn1])

        prelu1 = self.prelu1(merged0)
        conv2 = self.conv2(prelu1)
        bn3 = self.bn3(conv2)

        merged1 = self.merge1([bn3, merged0])

        prelu2 = self.prelu2(merged1)
        if self.atrous:
            aconv3 = self.aconv3(prelu2)
            bn4 = self.bn4(aconv3)
        else:
            conv3 = self.conv3(prelu2)
            bn4 = self.bn4(conv3)
        conv4 = self.conv4(inputs)

        return self.merge2([bn4, conv4])

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="res_conv_block1")
        return m.summary(**kwargs)


class Res_Convolution_Block2(Layer):
    """inspired by https://arxiv.org/pdf/2112.09654.pdf"""

    def __init__(
        self,
        kernels=16,
        size=3,
        activation="relu",
        level="",
        idx="",
        atrous=False,
        res_merge="max",
        **kwargs,
    ):
        super(Res_Convolution_Block2, self).__init__(**kwargs)

        self.atrous = atrous
        self.res_merge = normalize_merge(res_merge)

        self.prelu1 = PReLU(name="prelu1_" + str(level) + "_" + str(idx))
        self.prelu2 = PReLU(name="prelu2_" + str(level) + "_" + str(idx))
        self.prelu3 = PReLU(name="prelu3_" + str(level) + "_" + str(idx))
        self.prelu4 = PReLU(name="prelu4_" + str(level) + "_" + str(idx))

        # if self.res_merge == "max":
        #     self.merge1 = Maximum(name="max1_" + str(level) + "_" + str(idx))
        #     self.merge2 = Maximum(name="max2_" + str(level) + "_" + str(idx))
        #     self.merge3 = Maximum(name="max3_" + str(level) + "_" + str(idx))
        # elif self.res_merge == "concat":
        #     self.merge1 = Concatenate(name="concat1_" + str(level) + "_" + str(idx))
        #     self.merge2 = Concatenate(name="concat2_" + str(level) + "_" + str(idx))
        #     self.merge3 = Concatenate(name="concat3_" + str(level) + "_" + str(idx))
        # else:
        #     self.merge1 = Add(name="add1_" + str(level) + "_" + str(idx))
        #     self.merge2 = Add(name="add2_" + str(level) + "_" + str(idx))
        #     self.merge3 = Add(name="add3_" + str(level) + "_" + str(idx))

        self.merge1 = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="resconv2_merge1" + str(level) + "_" + str(idx),
        )
        self.merge2 = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="resconv2_merge2" + str(level) + "_" + str(idx),
        )
        self.merge3 = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="resconv2_merge3" + str(level) + "_" + str(idx),
        )

        self.add = Add(
            name="res_conv_block2_" + str(level) + "_" + str(idx) + "_output"
        )

        self.bn1 = BatchNormalization(name="bn1_" + str(level) + "_" + str(idx))
        self.bn2 = BatchNormalization(name="bn2_" + str(level) + "_" + str(idx))
        self.bn3 = BatchNormalization(name="bn3_" + str(level) + "_" + str(idx))
        self.bn4 = BatchNormalization(name="bn4_" + str(level) + "_" + str(idx))

        self.conv1 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv1_" + str(level) + "_" + str(idx),
        )
        self.conv2 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv2_" + str(level) + "_" + str(idx),
        )
        self.conv3 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv3_" + str(level) + "_" + str(idx),
        )

        if self.atrous:
            self.aconv4 = AtrousConvPyramid2D(
                dtype=tf.float32, name="aconv4_" + str(level) + "_" + str(idx)
            )
        else:
            self.conv4 = Conv2D(
                kernels,
                size,
                padding="same",
                activation=activation,
                dtype=tf.float32,
                name="conv4_" + str(level) + "_" + str(idx),
            )
        self.conv5 = Conv2D(
            kernels,
            1,
            padding="same",
            activation=activation,
            dtype=tf.float32,
            name="conv5_" + str(level) + "_" + str(idx),
        )

    def call(self, inputs):

        prelu1 = self.prelu1(inputs)
        conv1 = self.conv1(prelu1)
        bn1 = self.bn1(conv1)

        merged1 = self.merge1([bn1, conv1])

        prelu2 = self.prelu2(merged1)
        conv2 = self.conv2(prelu2)
        bn2 = self.bn2(conv2)

        merged2 = self.merge2([bn2, merged1])

        prelu3 = self.prelu3(merged2)
        conv3 = self.conv3(prelu3)
        bn3 = self.bn3(conv3)

        max3 = self.merge3([bn3, merged2])
        prelu4 = self.prelu4(max3)
        # added = self.add([bn3, merged2, merged1])
        # prelu4 = self.prelu4(added)

        if self.atrous:
            aconv4 = self.aconv4(prelu4)
            bn4 = self.bn4(aconv4)
        else:
            conv4 = self.conv4(prelu4)
            bn4 = self.bn4(conv4)

        conv5 = self.conv5(inputs)
        return self.add([bn4, conv5])

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="res_conv_block2")
        return m.summary(**kwargs)


class EncoderBlock(Layer):
    """based on Res_Convolution_Block1 and Res_Convolution_Block2"""

    def __init__(
        self,
        kernels=16,
        size=3,
        activation="relu",
        level=0,
        atrous=False,
        res_merge="max",
        **kwargs,
    ):

        super(EncoderBlock, self).__init__(**kwargs)
        self.level = level
        self.kernels = kernels
        self.size = size
        self.res_merge = res_merge
        self.atrous = atrous
        self.activation = activation

        if level:
            self.rconv1 = Res_Convolution_Block2(
                kernels=self.kernels,
                size=self.size,
                activation=self.activation,
                name="res_conv_encoder_block2_" + str(self.level) + "_" + str(0),
                level=self.level,
                idx=0,
                atrous=self.atrous,
                res_merge=self.res_merge,
                dtype=tf.float32,
            )

        else:
            self.rconv1 = Res_Convolution_Block1(
                kernels=self.kernels,
                size=self.size,
                activation=self.activation,
                name="res_conv_encoder_block1_" + str(self.level) + "_" + str(0),
                level=self.level,
                idx=0,
                atrous=self.atrous,
                res_merge=self.res_merge,
                dtype=tf.float32,
            )
        self.rconv2 = Res_Convolution_Block2(
            kernels=self.kernels,
            size=self.size,
            activation=self.activation,
            name="res_conv_encoder_block2_" + str(self.level) + "_" + str(1),
            level=self.level,
            idx=1,
            atrous=atrous,
            res_merge=self.res_merge,
            dtype=tf.float32,
        )

        # self.add = Add(name="encoder_" + str(level) + "_output", dtype=tf.float32)
        self.merge = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="encoder_" + str(level) + "_output",
        )

    def call(self, inputs):

        rconv1 = self.rconv1(inputs)
        rconv2 = self.rconv2(rconv1)
        merged = self.merge([rconv1, rconv2])

        return {"encoder_" + str(self.level) + "_output": merged}

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "kernels": self.kernels,
                "filter_size": self.size,
                "level": self.level,
            }
        )
        return cfg

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="encoder_block")
        return m.summary(**kwargs)


class BottleNeck(Layer):
    """based on Res_Convolution_Block1 and Res_Convolution_Block2"""

    def __init__(
        self, kernels=256, size=7, activation="relu", res_merge="max", **kwargs
    ):

        super(BottleNeck, self).__init__(**kwargs)
        self.kernels = kernels
        self.size = size
        self.res_merge = res_merge

        self.rconv1 = Res_Convolution_Block2(
            kernels=kernels,
            size=size,
            name="bottleneck_res_conv_block2_0",
            level="bottleneck",
            idx=0,
        )
        self.rconv2 = Res_Convolution_Block2(
            kernels=kernels,
            size=size,
            name="bottleneck_res_conv_block2_1",
            level="bottleneck",
            idx=1,
        )
        self.merge = Merge(
            kind=self.res_merge,
            activation="relu",
            out_channel=kernels,
            name="bottleneck_merged",
        )

    def call(self, inputs):

        rconv1 = self.rconv1(inputs)
        rconv2 = self.rconv2(rconv1)
        merged = self.merge([rconv1, rconv2])

        return {"bottleneck_output": merged}

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernels": self.kernels, "filter_size": self.size})
        return cfg

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="bottleneck_layer")
        return m.summary(**kwargs)


class AttentionGate(Layer):
    """inspired by https://arxiv.org/abs/1804.03999 (attention unet)"""

    def __init__(self, filters, level=0, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.level = level
        self.conv1 = Conv2D(
            self.filters,
            1,
            1,
            padding="same",
            name="gating_input_" + str(level),
            activation=None,
        )
        self.conv2 = Conv2D(
            self.filters,
            1,
            2,
            padding="same",
            name="skip_connection_input_" + str(level),
            activation=None,
        )

        self.conv3 = Conv2D(1, 1, 1, padding="same", name="added_input_" + str(level))
        self.add = Add(name="added_attention_input_" + str(level))
        self.relu = Activation("relu", name="attention_relu_" + str(level))
        self.sigmoid = Activation("sigmoid", name="sigmoid_" + str(level))
        self.upsample = UpSampling2D(
            2, interpolation="bilinear", name="upsampler_" + str(level)
        )
        self.multiply = Multiply(name="attention_gate_" + str(level) + "_output")

    def call(self, inputs):

        conv1 = self.conv1(inputs[0])
        conv2 = self.conv2(inputs[1])
        added = self.add([conv1, conv2])
        relu = self.relu(added)
        conv3 = self.conv3(relu)
        sigmoid = self.sigmoid(conv3)
        upsample = self.upsample(sigmoid)
        output = self.multiply([upsample, inputs[1]])

        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernels": self.filters})
        return cfg

    def summary(self, **kwargs):
        input1 = Input(shape=(64, 64, 64), name="gating_signal")
        input2 = Input(shape=(128, 128, 128), name="skip_connection_signal")
        m = Model(
            inputs=[input1, input2],
            outputs=self.call([input1, input2]),
            name="attention_gate_layer",
        )
        return m.summary(**kwargs)


class BoundaryAttentionGate(Layer):
    """inspired by https://arxiv.org/abs/1804.03999 (attention unet)"""

    def __init__(self, filters, level=0, **kwargs):
        super(BoundaryAttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.level = level
        self.conv1 = Conv2D(
            self.filters,
            1,
            1,
            padding="same",
            name="gating_input_" + str(level),
            activation=None,
        )
        self.conv2 = Conv2D(
            self.filters,
            1,
            2,
            padding="same",
            name="laplace_input_" + str(level),
            activation=None,
        )
        self.conv3 = Conv2D(
            self.filters,
            1,
            2,
            padding="same",
            name="skip_connection_input_" + str(level),
            activation=None,
        )
        self.conv4 = Conv2D(1, 1, 1, padding="same", name="added_input1_" + str(level))
        self.conv5 = Conv2D(1, 1, 1, padding="same", name="added_input2_" + str(level))

        self.add1 = Add(name="added_attention_input1_" + str(level))
        self.add2 = Add(name="added_attention_input2_" + str(level))
        self.relu1 = Activation("relu", name="attention_relu1_" + str(level))
        self.relu2 = Activation("relu", name="attention_relu2_" + str(level))
        self.sigmoid = Activation("sigmoid", name="sigmoid_" + str(level))
        self.upsample = UpSampling2D(
            2, interpolation="bilinear", name="upsampler_" + str(level)
        )
        self.multiply = Multiply(name="attention_gate_" + str(level) + "_output")

    def call(self, inputs):

        conv1 = self.conv1(inputs[0])
        conv2 = self.conv2(inputs[1])
        conv3 = self.conv3(inputs[2])

        added1 = self.add1([conv1, conv3])
        relu1 = self.relu1(added1)
        conv4 = self.conv4(relu1)

        added2 = self.add2([conv4, conv2])
        relu2 = self.relu2(added2)
        conv5 = self.conv5(relu2)
        sigmoid = self.sigmoid(conv5)
        upsample = self.upsample(sigmoid)

        output = self.multiply([upsample, inputs[2]])

        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernels": self.filters})
        return cfg

    def summary(self, **kwargs):
        input1 = Input(shape=(64, 64, 64), name="gating_signal")
        input2 = Input(shape=(128, 128, 128), name="skip_connection_signal")
        input3 = Input(shape=(128, 128, 128), name="skip_connection_signal")
        m = Model(
            inputs=[input1, input2],
            outputs=self.call([input1, input2, input3]),
            name="attention_gate_layer",
        )
        return m.summary(**kwargs)


class Vanilla_ResidualConvBlock(Layer):
    """vanilla residual convolution block
    inspired by https://arxiv.org/pdf/1512.03385.pdf"""

    def __init__(
        self,
        kernels=16,
        size=3,
        res_merge="sum",
        level="",
        idx="",
        **kwargs,
    ):
        super(Vanilla_ResidualConvBlock, self).__init__(**kwargs)

        self.relu1 = Activation("relu", name="relu1_" + str(level) + "_" + str(idx))
        self.relu2 = Activation("relu", name="relu2_" + str(level) + "_" + str(idx))

        # self.add = Add(name="add_" + str(level) + "_" + str(idx))
        self.merge = Merge(
            kind=res_merge,
            activation="relu",
            out_channel=kernels,
            name="merge_" + str(level) + "_" + str(idx),
        )

        self.bn1 = BatchNormalization(name="bn1_" + str(level) + "_" + str(idx))
        self.bn2 = BatchNormalization(name="bn2_" + str(level) + "_" + str(idx))

        self.conv1 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=None,
            dtype=tf.float32,
            name="conv1_" + str(level) + "_" + str(idx),
        )
        self.conv2 = Conv2D(
            kernels,
            size,
            padding="same",
            activation=None,
            dtype=tf.float32,
            name="conv2_" + str(level) + "_" + str(idx),
        )
        self.conv3 = Conv2D(
            kernels,
            1,
            padding="same",
            activation=None,
            dtype=tf.float32,
            name="conv3_" + str(level) + "_" + str(idx),
        )

    def call(self, inputs):

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        conv3 = self.conv3(inputs)

        merged = self.merge([bn2, conv3])
        relu2 = self.relu2(merged)

        return relu2

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(
            inputs=inputs, outputs=self.call(inputs), name="vanilla_res_conv_block2"
        )
        return m.summary(**kwargs)


class Vanilla_DecoderBlock(Layer):
    """based on Vanilla_ResidualConvBlock"""

    def __init__(self, kernels=128, size=7, merge="concat", level=0, **kwargs):
        super(Vanilla_DecoderBlock, self).__init__(**kwargs)
        self.merge = normalize_merge(merge)
        self.level = level
        self.kernels = kernels
        self.size = size
        self.conv = None
        self.rconv0 = Vanilla_ResidualConvBlock(
            kernels=kernels,
            size=size,
            level=self.level,
            idx=0,
            name="vanilla_res_conv_decoder_block_" + str(self.level) + "_" + str(0),
        )
        self.rconv1 = Vanilla_ResidualConvBlock(
            kernels=kernels,
            size=size,
            level=self.level,
            idx=1,
            name="vanilla_res_conv_decoder_block_" + str(self.level) + "_" + str(1),
        )
        self.rconv2 = Vanilla_ResidualConvBlock(
            kernels=kernels,
            size=size,
            level=self.level,
            idx=2,
            name="vanilla_res_conv_decoder_block_" + str(self.level) + "_" + str(2),
        )
        # self.rconv3 = Vanilla_ResidualConvBlock(
        #     kernels=kernels,
        #     size=size,
        #     level=self.level,
        #     idx=3,
        #     name="vanilla_res_conv_decoder_block_" + str(self.level) + "_" + str(3),
        # )
        self.add0 = Add(name="vanilla_res_conv_decoder_added" + str(self.level))
        # self.add1 = Add(name="vanilla_res_conv_decoder_" + str(self.level) + "_output")

        self.merge = Merge(
            kind=self.merge,
            activation="relu",
            out_channel=kernels,
            name="decoder_merge" + str(level),
        )
        # if self.merge == "max":
        #     self.max = Maximum(name="merge_max_" + str(self.level))
        # elif self.merge == "concat":
        #     self.concat = Concatenate(name="merge_concat_" + str(self.level))
        # else:
        #     self.summed = Add(name="merge_sum_" + str(self.level))

    def call(self, inputs):

        if self.merge == "max" or self.merge == "sum":
            filters = inputs[-1].shape[-1]

            if self.conv == None:
                self.conv = Conv2D(
                    filters, 1, 1, padding="same", name="conv_backup_" + str(self.level)
                )
            conv = self.conv(inputs[0])
            merged = self.merge([conv, inputs[-1]])
            # try:
            #     merged = self.max(inputs)
            # except ValueError:
            #     raise ValueError(
            #         f"Can't use maxout to the inputs of {self.max.name} layer in \
            #             {self.name} due to unequal channels. Either change the number \
            #                 of filters in the previous deconv layer or use an intermediate conv layer"
            #     )
        else:
            merged = self.merge(inputs)
        # else:
        #     try:
        #         merged = self.summed(inputs)
        #     except ValueError:
        #         raise ValueError(
        #             f"Can't add the elements of inputs of {self.summed.name} layer in\
        #                 {self.name} due to unequal channels. Either change the number \
        #                     of filters in the previous deconv layer or use an intermediate conv layer"
        #         )

        rconv0 = self.rconv0(merged)
        rconv1 = self.rconv1(rconv0)
        added0 = self.add0([rconv1, rconv0])

        rconv2 = self.rconv2(added0)
        # rconv3 = self.rconv3(rconv2)
        # added1 = self.add1([rconv3, rconv2])

        return {"decoder_" + str(self.level) + "_output": rconv2}

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "kernels": self.kernels,
                "filter_size": self.size,
                "level": self.level,
                "merge_type": self.merge,
            }
        )
        return cfg

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs1 = Input(shape=input_shape)
        inputs2 = Input(shape=input_shape)
        m = Model(
            inputs=[inputs1, inputs2],
            outputs=self.call([inputs1, inputs2]),
            name="decoder_block",
        )
        return m.summary(**kwargs)


class BoundaryBlock(Layer):
    def __init__(self, **kwargs):
        super(BoundaryBlock, self).__init__(**kwargs)

    def call(self, inputs):
        min_pool1 = tf.nn.max_pool2d(-inputs, 3, 1, "SAME") * -1
        max_min_pool1 = tf.nn.max_pool2d(min_pool1, 3, 1, "SAME")
        boundary1 = tf.nn.relu(max_min_pool1 - min_pool1)
        boundary1 = tf.reduce_sum(boundary1, axis=-1, keepdims=True)

        # min_pool2 = tf.nn.max_pool2d(-inputs, 3, 2, "VALID") * -1
        # max_min_pool2 = tf.nn.max_pool2d(min_pool2, 3, 2, "VALID")
        # boundary2 = tf.nn.relu(max_min_pool2 - min_pool2)
        # boundary2 = tf.reduce_sum(boundary2, axis=-1, keepdims=True)

        # min_pool3 = tf.nn.max_pool2d(-inputs, 3, 3, "VALID") * -1
        # max_min_pool3 = tf.nn.max_pool2d(min_pool3, 3, 3, "VALID")
        # boundary3 = tf.nn.relu(max_min_pool3 - min_pool3)
        # boundary3 = tf.reduce_sum(boundary3, axis=-1, keepdims=True)

        return boundary1


class AGFFBlock(Layer):
    """inspired by https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9410460"""

    def __init__(self, kernels=64, lower_in_channel=32, **kwargs):
        super(AGFFBlock, self).__init__(**kwargs)
        self.gap = GlobalAveragePooling2D(keepdims=True, name="gap")
        self.conv1 = Conv2D(
            kernels, 1, 1, padding="same", name="conv1", activation="relu"
        )
        self.conv2 = Conv2D(
            lower_in_channel, 1, 1, padding="same", name="conv2", activation="sigmoid"
        )
        self.mult = Multiply(name="agff_output")

    def call(self, inputs):

        gap = self.gap(inputs[1])
        conv1 = self.conv1(gap)
        conv2 = self.conv2(conv1)

        return self.mult([inputs[0], conv2])

    def summary(self, **kwargs):
        inputs1 = Input(shape=(256, 256, 32))
        inputs2 = Input(shape=(128, 128, 64))
        m = Model(
            inputs=[inputs1, inputs2],
            outputs=self.call([inputs1, inputs2]),
            name="AGFF_block",
        )
        return m.summary(**kwargs)


class SEBlock(Layer):
    """inspired by https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(self, downsample_ratio=2, in_channel=256, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.gap = GlobalAveragePooling2D(keepdims=True, name="gap")
        self.conv1 = Conv2D(
            in_channel / downsample_ratio,
            1,
            padding="same",
            name="conv1",
            activation="relu",
        )
        self.conv2 = Conv2D(
            in_channel, 1, padding="same", name="conv2", activation="sigmoid"
        )
        self.mult = Multiply(name="se_output")

    def call(self, inputs):

        gap = self.gap(inputs)
        conv1 = self.conv1(gap)
        conv2 = self.conv2(conv1)

        return self.mult([inputs, conv2])

    def summary(self, **kwargs):
        inputs = Input(shape=(256, 256, 32))
        m = Model(
            inputs=inputs,
            outputs=self.call(inputs),
            name="SE_block",
        )
        return m.summary(**kwargs)


class BAGMBlock(Layer):
    """inspired by https://ieeexplore.ieee.org/ielx7/9411940/9411911/09412621.pdf"""

    def __init__(self, in_channel=256, level=3, **kwargs):

        super(BAGMBlock, self).__init__(**kwargs)
        self.level = level
        self.agff = AGFFBlock(name="agff", lower_in_channel=in_channel)
        self.conv_s1 = Conv2D(in_channel // 2, 3, 1, name="conv_seg1", padding="same")
        self.conv_s2 = Conv2D(in_channel // 2, 1, 1, name="conv_seg2")
        self.conv_b1 = Conv2D(in_channel // 2, 3, 1, name="conv_bound1", padding="same")
        self.conv_b2 = Conv2D(in_channel // 2, 1, 1, name="conv_bound2")
        self.conv_b11 = Conv2D(
            in_channel, 1, activation="softmax", name="attention_weights"
        )
        self.concat = Concatenate(name="merge_concat")
        self.se = SEBlock(in_channel=in_channel)
        self.mult = Multiply(name="scaled_map")
        self.add = Add(name="bagm_output")

    def call(self, inputs):
        agff = self.agff(inputs[:2])
        conv_s1 = self.conv_s1(agff)
        conv_s2 = self.conv_s2(conv_s1)
        conv_b1 = self.conv_b1(inputs[-1])
        conv_b2 = self.conv_b2(conv_b1)

        concat = self.concat([conv_s2, conv_b2])
        se = self.se(concat)
        conv_b11 = self.conv_b11(inputs[-1])
        mult = self.mult([se, conv_b11])

        return {"bagm_" + str(self.level) + "_output": self.add([se, mult])}

    def summary(self, **kwargs):
        inputs1 = Input(shape=(40, 32, 256))
        inputs2 = Input(shape=(20, 16, 512))
        inputs3 = Input(shape=(40, 32, 256))
        m = Model(
            inputs=[inputs1, inputs2, inputs3],
            outputs=self.call([inputs1, inputs2, inputs3]),
            name="BAGM_block",
        )
        return m.summary(**kwargs)


class ResidualGlobalConv2D(Layer):
    def __init__(self, inp_shape, filters, merge="sum", **kwargs):

        super(ResidualGlobalConv2D, self).__init__(**kwargs)
        self.inp_shape = inp_shape
        self.filters = filters
        self.merge = merge

        self.resizing = Resizing(
            self.inp_shape[0],
            self.inp_shape[1],
            interpolation="bilinear",
            name="laplace_resizing",
        )
        self.globconv = GlobalConv2D(
            kernels=self.filters,
            kernel_param=5,
            merge="sum",
            padding="same",
            name="laplace_global_conv",
        )
        self.resconv = Vanilla_ResidualConvBlock(
            kernels=self.filters, name="laplace_residual"
        )
        self.conv = Conv2D(
            self.filters,
            1,
            1,
            padding="same",
            activation="sigmoid",
            name="laplace_conv",
        )

        if self.merge == "sum":
            self.add = Add(name="laplace_sum")
        else:
            self.concat = Concatenate(name="laplace_concat")

    def call(self, inputs):

        resized = self.resizing(inputs)
        globconv = self.globconv(resized)
        resconv = self.resconv(resized)

        if self.merge == "sum":
            merged_output = self.add([globconv, resconv])
        else:
            merged_output = self.concat([globconv, resconv])

        return self.conv(merged_output)


class ConvBlock2D(Layer):
    def __init__(self, out_channels=256, **kwargs):

        self.out_channels = out_channels
        self.ksizes = [1, 3, 3, 3, 1]
        self.kernels = [256, 256, 256, 256, self.out_channels]

        self.convs = [
            Conv2D(kernel, ksize, padding="same", activation="relu")
            for kernel, ksize in zip(self.kernels, self.ksizes)
        ]
        self.bns = [BatchNormalization() for i, _ in enumerate(self.ksizes)]

    def call(self, inputs):
        x = inputs

        for i, _ in enumerate(self.ksizes):
            x = self.convs[i](x)
            x = self.bns[i](x)

        return x
