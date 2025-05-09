import math
import numpy as np
import tensorflow as tf
from collections.abc import Iterable
from tensorflow.keras.layers import (
    Conv2D,
    PReLU,
    Lambda,
    Layer,
    InputSpec,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    UpSampling2D,
    Maximum,
    Concatenate,
    Input,
    Add,
    Reshape,
    Embedding,
    BatchNormalization,
)
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from .utils import normalize_tuple, normalize_data_format, normalize_merge
from .utils import Pooling_Op
from .AtrousConv2D import AtrousConvPyramid2D
from .SpatialPyramidPooling2D import SpatialPyramidPooling2D
from ...config import NUM_CLASS, INPUT_SHAPE, BATCH_SIZE

# from ..layers import ArgMax, OneHot


class Res_Convolution_Block1(Layer):
    def __init__(
        self,
        kernels=16,
        size=3,
        activation="relu",
        level="",
        idx="",
        atrous=False,
        **kwargs,
    ):
        super(Res_Convolution_Block1, self).__init__(**kwargs)

        self.atrous = atrous

        self.prelu1 = PReLU(name="prelu1_" + str(level) + "_" + str(idx))
        self.prelu2 = PReLU(name="prelu2_" + str(level) + "_" + str(idx))
        self.prelu3 = PReLU(name="prelu3_" + str(level) + "_" + str(idx))

        self.maxout1 = Maximum(name="max1_" + str(level) + "_" + str(idx))
        self.maxout2 = Maximum(name="max2_" + str(level) + "_" + str(idx))

        self.bn1 = BatchNormalization(name="bn1_" + str(level) + "_" + str(idx))
        self.bn2 = BatchNormalization(name="bn2_" + str(level) + "_" + str(idx))
        self.bn3 = BatchNormalization(name="bn3_" + str(level) + "_" + str(idx))
        self.bn4 = BatchNormalization(name="bn4_" + str(level) + "_" + str(idx))
        self.bn5 = BatchNormalization(name="bn5_" + str(level) + "_" + str(idx))

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
                # dilation_rates=[2, 4, 6],
                dtype=tf.float32,
                name="aconv4_" + str(level) + "_" + str(idx),
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

    def call(self, inputs):

        bn1 = self.bn1(inputs)
        conv1 = self.conv1(bn1)
        bn2 = self.bn2(conv1)

        prelu1 = self.prelu1(bn2)
        conv2 = self.conv2(prelu1)
        bn3 = self.bn3(conv2)

        max1 = self.maxout1([bn3, bn2])

        prelu2 = self.prelu2(max1)
        conv3 = self.conv3(prelu2)
        bn4 = self.bn4(conv3)

        max2 = self.maxout2([bn4, max1])

        prelu3 = self.prelu3(max2)
        if self.atrous:
            conv4 = self.aconv4(prelu3)
        else:
            conv4 = self.conv4(prelu3)
        bn5 = self.bn5(conv4)

        return bn5

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="res_conv_block1")
        return m.summary(**kwargs)


class Res_Convolution_Block2(Layer):
    def __init__(
        self,
        kernels=16,
        size=3,
        activation="relu",
        level="",
        idx="",
        atrous=False,
        **kwargs,
    ):
        super(Res_Convolution_Block2, self).__init__(**kwargs)

        self.atrous = atrous

        self.prelu1 = PReLU(name="prelu1_" + str(level) + "_" + str(idx))
        self.prelu2 = PReLU(name="prelu2_" + str(level) + "_" + str(idx))
        self.prelu3 = PReLU(name="prelu3_" + str(level) + "_" + str(idx))
        self.prelu4 = PReLU(name="prelu4_" + str(level) + "_" + str(idx))

        self.maxout1 = Maximum(name="max1_" + str(level) + "_" + str(idx))
        self.maxout2 = Maximum(name="max2_" + str(level) + "_" + str(idx))
        self.maxout3 = Maximum(name="max3_" + str(level) + "_" + str(idx))

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

    def call(self, inputs):

        prelu1 = self.prelu1(inputs)
        conv1 = self.conv1(prelu1)
        bn1 = self.bn1(conv1)

        max1 = self.maxout1([bn1, conv1])

        prelu2 = self.prelu2(max1)
        conv2 = self.conv2(prelu2)
        bn2 = self.bn2(conv2)

        max2 = self.maxout2([bn2, max1])

        prelu3 = self.prelu3(max2)
        conv3 = self.conv3(prelu3)
        bn3 = self.bn3(conv3)

        max3 = self.maxout3([bn3, max2, max1])

        prelu4 = self.prelu4(max3)
        if self.atrous:
            conv4 = self.aconv4(prelu4)
        else:
            conv4 = self.conv4(prelu4)
        bn4 = self.bn4(conv4)

        return bn4

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="res_conv_block2")
        return m.summary(**kwargs)


class EncoderBlock(Layer):
    def __init__(
        self,
        kernels=16,
        size=3,
        activation="relu",
        level=0,
        atrous=False,
        SPP=False,
        **kwargs,
    ):

        super(EncoderBlock, self).__init__(**kwargs)
        if level:
            if SPP:
                self.sppool = SpatialPyramidPooling2D(
                    [100, 150, 200], name="spp" + str(level)
                )
            else:
                self.maxpool = MaxPooling2D(
                    name="maxpool_" + str(level), dtype=tf.float32
                )

            self.rconv1 = Res_Convolution_Block2(
                kernels=kernels,
                size=size,
                activation=activation,
                name="res_conv_block_" + str(level) + "_" + str(0),
                level=level,
                idx=0,
                dtype=tf.float32,
            )

        else:
            self.rconv1 = Res_Convolution_Block1(
                kernels=kernels,
                size=size,
                activation=activation,
                name="res_conv_block_" + str(level) + "_" + str(0),
                level=level,
                idx=0,
                dtype=tf.float32,
            )
        self.rconv2 = Res_Convolution_Block2(
            kernels=kernels,
            size=size,
            activation=activation,
            name="res_conv_block_" + str(level) + "_" + str(1),
            level=level,
            idx=1,
            atrous=atrous,
            dtype=tf.float32,
        )

        self.add = Add(name="addition_" + str(level), dtype=tf.float32)
        self.level = level
        self.kernels = kernels
        self.size = size
        self.spp = SPP

    def call(self, inputs):

        if self.level:
            if self.spp:
                pooled = self.sppool(inputs)
            else:
                pooled = self.maxpool(inputs)
            rconv1 = self.rconv1(pooled)
        else:
            rconv1 = self.rconv1(inputs)

        rconv2 = self.rconv2(rconv1)
        added = self.add([rconv1, rconv2])

        return {"encoder_out_" + str(self.level): added}

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


class DecoderBlock(Layer):
    def __init__(self, kernels=128, size=7, merge="max", level=0, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.merge = normalize_merge(merge)
        self.level = level
        self.kernels = kernels
        self.size = size
        self.rconv1 = Res_Convolution_Block2(
            kernels=kernels,
            size=size,
            name="res_conv_decoder_block_" + str(level) + "_" + str(0),
        )
        self.rconv2 = Res_Convolution_Block2(
            kernels=kernels,
            size=size,
            name="res_conv_decoder_block_" + str(level) + "_" + str(1),
        )
        if self.level:
            self.upsample = UpSampling2D(
                (2, 2), interpolation="lanczos3", name="upsample_" + str(level)
            )
            self.conv1 = Conv2D(kernels // 2, 1, activation="relu")
            self.add = Add(name="added_" + str(level))
        else:
            # self.argmax = Lambda(
            #     lambda x: tf.expand_dims(tf.math.argmax(x, -1), -1), name="mask"
            # )
            self.conv1 = Conv2D(
                NUM_CLASS,
                1,
                activation=tf.keras.activations.softmax,
                name="probability_map",
            )
            self.conv2 = Conv2D(
                NUM_CLASS,
                1,
                activation=tf.keras.activations.softmax,
                name="mask_one_hot",
            )
            self.conv3 = Conv2D(1, 1, activation=None, name="mask")

        if self.merge == "max":
            self.max = Maximum(name="merge_max_" + str(level))
        elif self.merge == "concat":
            self.concat = Concatenate(name="merge_concat_" + str(level))
        else:
            self.summed = Add(name="merge_sum_" + str(level))

    def call(self, inputs):

        if self.merge == "max":
            merged = self.max(inputs)
        elif self.merge == "concat":
            merged = self.concat(inputs)
        else:
            merged = self.summed(inputs)

        rconv1 = self.rconv1(merged)
        rconv2 = self.rconv2(rconv1)

        if self.level:
            added = self.add([rconv2, rconv1])
            upsample = self.upsample(added)
            upsampled = self.conv1(upsample)
            return {"decoder_out_" + str(self.level): upsampled}
        else:
            conv1 = self.conv1(rconv2)
            conv2 = self.conv2(rconv2)
            conv3 = self.conv3(conv2)

            # argmaxed = self.argmax(conv2)
            # one_hot = self.one_hot(conv1)

            return {"mask": conv3, "mask_one_hot": conv2, "probability_map": conv1}

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


class BottleNeck(Layer):
    def __init__(self, kernels=256, size=7, activation="relu", **kwargs):

        super(BottleNeck, self).__init__(**kwargs)
        self.maxpool = MaxPooling2D(name="bottleneck_maxpool", dtype=tf.float32)
        self.rconv1 = Res_Convolution_Block2(
            kernels=kernels,
            size=size,
            name="bottleneck_res_conv_0",
            level="bottleneck",
            idx=0,
        )
        self.rconv2 = Res_Convolution_Block2(
            kernels=kernels,
            size=size,
            name="bottleneck_res_conv_1",
            level="bottleneck",
            idx=1,
        )
        self.convTr = Conv2DTranspose(
            kernels // 2,
            2,
            (2, 2),
            name="bottleneck_convTr",
            activation=activation,
            dtype=tf.float32,
        )
        self.kernels = kernels
        self.size = size

    def call(self, inputs):

        maxpool = self.maxpool(inputs)
        rconv1 = self.rconv1(maxpool)
        rconv2 = self.rconv2(rconv1)
        convTr = self.convTr(rconv2)

        return {"bottleneck_out": convTr}

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kernels": self.kernels, "filter_size": self.size})
        return cfg

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="bottleneck_layer")
        return m.summary(**kwargs)
