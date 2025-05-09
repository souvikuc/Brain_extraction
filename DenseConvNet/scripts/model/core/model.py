import numpy as np
import os, glob, h5py, sys, json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import (
    Add,
    Input,
    Lambda,
    Conv2D,
    Average,
    Resizing,
    Multiply,
    MaxPooling2D,
    Conv2DTranspose,
)

from ....config import BaseModel
from ..dataloader.datagen import DataLoader
from .utils.layers import MixedPooling2D
from ....file_utils.config_utils import read_configs
from .utils.metrics import (
    Multiclass_Weighted_SoftMax_DiceSimilarity,
    Batchwise_Avg_DiceSimilarity,
)
from .utils.utils import (
    normalize_merge,
    normalize_pooling,
    normalize_tuple,
    normalize_unpooling,
)
from .utils.blocks import (
    Merge,
    BAGMBlock,
    BottleNeck,
    EncoderBlock,
    BoundaryBlock,
    AttentionGate,
    BoundaryAttentionGate,
    Vanilla_DecoderBlock,
    # ResidualGlobalConv2D,
    SpatialPyramidPooling2D,
)
from .utils.losses import (
    Multiclass_Weighted_CE,
    Pixelwise_Weighted_CE,
    Multiclass_Weighted_FocalLoss,
    Pixelwise_Weighted_FocalLoss,
    Multi_Weighted_SoftMax_DiceLoss,
)

configs = read_configs()

EPSILON = configs["EPSILON"]
# DATA_DIR = configs["DATA_DIR"]
PLANE_IDX = configs["PLANE_IDX"]
MODEL_DIR = configs["MODEL_DIR"]
NUM_CLASS = configs["NUM_CLASS"]
BATCH_SIZE = configs["BATCH_SIZE"]
INPUT_SHAPE = configs["INPUT_SHAPE"]
STEPS_PER_EPOCH = configs["STEPS_PER_EPOCH"]


seg_d_sim1 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    name="seg_all_class_dice_similarity"
)
seg_d_sim2 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    include_class=[1, 2], name="seg_selected_dice_similarity"
)
seg_d_sim3 = Batchwise_Avg_DiceSimilarity()

fusion_d_sim1 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    name="fusion_all_class_dice_similarity"
)
fusion_d_sim2 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    include_class=[1, 2], name="fusion_selected_dice_similarity"
)
fusion_d_sim3 = Batchwise_Avg_DiceSimilarity()

final_d_sim1 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    name="final_all_class_dice_similarity"
)
final_d_sim2 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    include_class=[1, 2], name="final_selected_dice_similarity"
)
final_d_sim3 = Batchwise_Avg_DiceSimilarity()

bound_d_sim1 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    name="bound_all_class_dice_similarity"
)
bound_d_sim2 = Multiclass_Weighted_SoftMax_DiceSimilarity(
    include_class=[1], name="bound_selected_dice_similarity"
)
bound_d_sim3 = Batchwise_Avg_DiceSimilarity()


class ResDenseConvNet(Model):
    def __init__(
        self,
        pooling="mixed",
        atrous=False,
        skip_connection="concat",
        plane="sagittal",
        **kwargs
    ):

        super(ResDenseConvNet, self).__init__(**kwargs)
        self.atrous = atrous
        self.plane = plane
        self.pooling = normalize_pooling(pooling)
        self.skip = normalize_merge(skip_connection)

        # ------------------------------------------------------------------------------
        # Encoder branch
        # ------------------------------------------------------------------------------
        self.encoder0 = EncoderBlock(
            kernels=32,
            size=3,
            level=0,
            res_merge="max",
            activation="relu",
            name="EncoderBlock0",
            dtype=tf.float32,
            atrous=self.atrous,
        )
        self.encoder1 = EncoderBlock(
            kernels=64,
            size=3,
            level=1,
            res_merge="max",
            activation="relu",
            name="EncoderBlock1",
            dtype=tf.float32,
            atrous=self.atrous,
        )
        self.encoder2 = EncoderBlock(
            kernels=128,
            size=3,
            level=2,
            res_merge="max",
            activation="relu",
            name="EncoderBlock2",
            dtype=tf.float32,
            atrous=self.atrous,
        )
        self.encoder3 = EncoderBlock(
            kernels=256,
            size=3,
            level=3,
            res_merge="max",
            activation="relu",
            name="EncoderBlock3",
            dtype=tf.float32,
            atrous=self.atrous,
        )
        if self.plane == "sagittal":
            self.enc_padding = Resizing(40, 32, name="encoder3_padding")
        elif self.plane == "coronal":
            self.enc_padding = Lambda(lambda x: x, name="encoder3_padding")
        else:
            self.enc_padding = Resizing(32, 40, name="encoder3_padding")

        self.bottleneck = BottleNeck(
            kernels=512, size=3, name="BottleNeck", dtype=tf.float32
        )

        if self.pooling == "max":
            self.enc_pool0 = MaxPooling2D(name="encoder_maxpool0")
            self.enc_pool1 = MaxPooling2D(name="encoder_maxpool1")
            self.enc_pool2 = MaxPooling2D(name="encoder_maxpool2")
            self.enc_pool3 = MaxPooling2D(name="encoder_maxpool3")
        elif self.pooling == "mixed":
            self.enc_pool0 = MixedPooling2D(name="encoder_mixed_pool0")
            self.enc_pool1 = MixedPooling2D(name="encoder_mixed_pool1")
            self.enc_pool2 = MixedPooling2D(name="encoder_mixed_pool2")
            self.enc_pool3 = MixedPooling2D(name="encoder_mixed_pool3")
        else:
            self.enc_pool0 = SpatialPyramidPooling2D(
                [50, 100, 150, 200], name="encoder_spp0"
            )
            self.enc_pool1 = SpatialPyramidPooling2D([40, 80, 100], name="encoder_spp1")
            self.enc_pool2 = SpatialPyramidPooling2D([20, 40, 60], name="encoder_spp2")
            self.enc_pool3 = SpatialPyramidPooling2D([10, 20], name="encoder_spp3")

        # ------------------------------------------------------------------------------
        # Laplace encoder branch
        # ------------------------------------------------------------------------------

        self.lap_encoder0 = EncoderBlock(
            kernels=32,
            size=3,
            level=0,
            res_merge="max",
            activation="relu",
            name="Laplace_EncoderBlock0",
            dtype=tf.float32,
            atrous=self.atrous,
        )
        self.lap_encoder1 = EncoderBlock(
            kernels=64,
            size=3,
            level=1,
            res_merge="max",
            activation="relu",
            name="Laplace_EncoderBlock1",
            dtype=tf.float32,
            atrous=self.atrous,
        )
        self.lap_encoder2 = EncoderBlock(
            kernels=128,
            size=3,
            level=2,
            res_merge="max",
            activation="relu",
            name="Laplace_EncoderBlock2",
            dtype=tf.float32,
            atrous=self.atrous,
        )
        self.lap_encoder3 = EncoderBlock(
            kernels=256,
            size=3,
            level=3,
            res_merge="max",
            activation="relu",
            name="Laplace_EncoderBlock3",
            dtype=tf.float32,
            atrous=self.atrous,
        )

        self.lap_conv0 = Conv2D(
            64, 1, 1, padding="same", activation="sigmoid", name="boundary_attention0"
        )
        self.lap_conv1 = Conv2D(
            128, 1, 1, padding="same", activation="sigmoid", name="boundary_attention1"
        )
        self.lap_conv2 = Conv2D(
            256, 1, 1, padding="same", activation="sigmoid", name="boundary_attention2"
        )
        self.lap_conv3 = Conv2D(
            512, 1, 1, padding="same", activation="sigmoid", name="boundary_attention3"
        )

        if self.plane == "sagittal":
            self.lap_enc_padding = Resizing(40, 32, name="laplace_encoder3_padding")
        elif self.plane == "coronal":
            self.lap_enc_padding = Lambda(lambda x: x, name="laplace_encoder3_padding")
        else:
            self.lap_enc_padding = Resizing(32, 40, name="laplace_encoder3_padding")

        if self.pooling == "max":
            self.lap_enc_pool0 = MaxPooling2D(name="laplace_encoder_maxpool0")
            self.lap_enc_pool1 = MaxPooling2D(name="laplace_encoder_maxpool1")
            self.lap_enc_pool2 = MaxPooling2D(name="laplace_encoder_maxpool2")
        elif self.pooling == "mixed":
            self.lap_enc_pool0 = MixedPooling2D(name="laplace_encoder_mixed_pool0")
            self.lap_enc_pool1 = MixedPooling2D(name="laplace_encoder_mixed_pool1")
            self.lap_enc_pool2 = MixedPooling2D(name="laplace_encoder_mixed_pool2")
        else:
            self.lap_enc_pool0 = SpatialPyramidPooling2D(
                [50, 100, 150, 200], name="laplace_encoder_spp0"
            )
            self.lap_enc_pool1 = SpatialPyramidPooling2D(
                [40, 80, 100], name="laplace_encoder_spp1"
            )
            self.lap_enc_pool2 = SpatialPyramidPooling2D(
                [20, 40, 60], name="laplace_encoder_spp2"
            )

        # ------------------------------------------------------------------------------
        # Skip connections branch
        # ------------------------------------------------------------------------------
        self.skip_conv01 = Conv2D(
            64, 3, padding="same", activation="relu", name="skip_conv_01"
        )
        self.skip_conv12 = Conv2D(
            128, 3, padding="same", activation="relu", name="skip_conv_12"
        )
        self.skip_conv23 = Conv2D(
            256, 3, padding="same", activation="relu", name="skip_conv_23"
        )

        if self.plane == "sagittal":

            self.skip_padding = Resizing(40, 32, name="skip_spp23_padding")
            self.skip_pooling01 = SpatialPyramidPooling2D(
                [50, 100, 150, 200], output_shape=(156, 128), name="skip_spp_01"
            )
            self.skip_pooling12 = SpatialPyramidPooling2D(
                [40, 80, 100], output_shape=(78, 64), name="skip_spp_12"
            )
            self.skip_pooling23 = SpatialPyramidPooling2D(
                [20, 40, 60], output_shape=(39, 32), name="skip_spp_23"
            )
        elif self.plane == "coronal":
            self.skip_padding = Lambda(lambda x: x, name="skip_spp23_padding")
            self.skip_pooling01 = SpatialPyramidPooling2D(
                [50, 100, 150, 200], output_shape=128, name="skip_spp_01"
            )
            self.skip_pooling12 = SpatialPyramidPooling2D(
                [40, 80, 100], output_shape=64, name="skip_spp_12"
            )
            self.skip_pooling23 = SpatialPyramidPooling2D(
                [20, 40, 60], output_shape=32, name="skip_spp_23"
            )
        else:
            self.skip_padding = Resizing(32, 40, name="skip_spp23_padding")
            self.skip_pooling01 = SpatialPyramidPooling2D(
                [50, 100, 150, 200], output_shape=(128, 156), name="skip_spp_01"
            )
            self.skip_pooling12 = SpatialPyramidPooling2D(
                [40, 80, 100], output_shape=(64, 78), name="skip_spp_12"
            )
            self.skip_pooling23 = SpatialPyramidPooling2D(
                [20, 40, 60], output_shape=(32, 39), name="skip_spp_23"
            )

        self.skip_add01 = Add(name="skip_add_01")
        self.skip_add12 = Add(name="skip_add_12")
        self.skip_add23 = Add(name="skip_add_23")

        # self.attn0 = BoundaryAttentionGate(32, level=0, name="boundary_attention_gate0")
        # self.attn1 = BoundaryAttentionGate(64, level=1, name="boundary_attention_gate1")
        # self.attn2 = BoundaryAttentionGate(
        #     128, level=2, name="boundary_attention_gate2"
        # )
        # self.attn3 = BoundaryAttentionGate(
        #     256, level=3, name="boundary_attention_gate3"
        # )

        self.attn0 = AttentionGate(32, level=0, name="attention_gate0")
        self.attn1 = AttentionGate(64, level=1, name="attention_gate1")
        self.attn2 = AttentionGate(128, level=2, name="attention_gate2")
        self.attn3 = AttentionGate(256, level=3, name="attention_gate3")

        # ------------------------------------------------------------------------------
        # Segmentation decoder branch
        # ------------------------------------------------------------------------------
        self.seg_decoder3 = Vanilla_DecoderBlock(
            kernels=256,
            size=3,
            merge=self.skip,
            level=3,
            name="segmentation_DecoderBlock3",
            dtype=tf.float32,
        )
        self.seg_decoder2 = Vanilla_DecoderBlock(
            kernels=128,
            size=3,
            merge=self.skip,
            level=2,
            name="segmentation_DecoderBlock2",
            dtype=tf.float32,
        )
        self.seg_decoder1 = Vanilla_DecoderBlock(
            kernels=64,
            size=3,
            merge=self.skip,
            level=1,
            name="segmentation_DecoderBlock1",
            dtype=tf.float32,
        )
        self.seg_decoder0 = Vanilla_DecoderBlock(
            kernels=32,
            size=3,
            merge=self.skip,
            level=0,
            name="segmentation_DecoderBlock0",
            dtype=tf.float32,
        )
        if self.plane == "sagittal":
            self.seg_resizing = Resizing(39, 32, name="segmentation_resizing")
        elif self.plane == "coronal":
            self.seg_resizing = Lambda(lambda x: x, name="segmentation_resizing")
        else:
            self.seg_resizing = Resizing(32, 39, name="segmentation_resizing")

        self.seg_unpool3 = Conv2DTranspose(512, 2, 2, name="segmentation_deconv3")
        self.seg_unpool2 = Conv2DTranspose(256, 2, 2, name="segmentation_deconv2")
        self.seg_unpool1 = Conv2DTranspose(128, 2, 2, name="segmentation_deconv1")
        self.seg_unpool0 = Conv2DTranspose(64, 2, 2, name="segmentation_deconv0")

        # self.seg_conv1 = Conv2D(
        #     64, 1, padding="same", activation=None, name="segmentation_conv"
        # )

        self.seg_conv2 = Conv2D(
            NUM_CLASS,
            1,
            padding="same",
            name="segmentation_probability_map",
            activation=tf.keras.activations.softmax,
        )
        self.seg_boundary_mask = BoundaryBlock(name="segmentation_boundary_mask")
        self.seg_segmentation_mask = Lambda(
            lambda x: tf.expand_dims(tf.math.argmax(x, -1), -1),
            name="seg_segmentation_mask",
            trainable=False,
            dtype=tf.float32,
        )

        # ------------------------------------------------------------------------------
        # Boundary decoder branch
        # ------------------------------------------------------------------------------

        self.bound_mult0 = Multiply(name="boundary_gate_attention0")
        self.bound_mult1 = Multiply(name="boundary_gate_attention1")
        self.bound_mult2 = Multiply(name="boundary_gate_attention2")
        self.bound_mult3 = Multiply(name="boundary_gate_attention3")

        self.bound_decoder3 = Vanilla_DecoderBlock(
            kernels=256,
            size=3,
            merge=self.skip,
            level=3,
            name="boundary_DecoderBlock3",
            dtype=tf.float32,
        )
        self.bound_decoder2 = Vanilla_DecoderBlock(
            kernels=128,
            size=3,
            merge=self.skip,
            level=2,
            name="boundary_DecoderBlock2",
            dtype=tf.float32,
        )
        self.bound_decoder1 = Vanilla_DecoderBlock(
            kernels=64,
            size=3,
            merge=self.skip,
            level=1,
            name="boundary_DecoderBlock1",
            dtype=tf.float32,
        )
        self.bound_decoder0 = Vanilla_DecoderBlock(
            kernels=32,
            size=3,
            merge=self.skip,
            level=0,
            name="boundary_DecoderBlock0",
            dtype=tf.float32,
        )
        if self.plane == "sagittal":
            self.bound_resizing = Resizing(39, 32, name="boundary_resizing")
        elif self.plane == "coronal":
            self.bound_resizing = Lambda(lambda x: x, name="boundary_resizing")
        else:

            self.bound_resizing = Resizing(32, 39, name="boundary_resizing")

        self.bound_unpool3 = Conv2DTranspose(512, 2, 2, name="boundary_deconv3")
        self.bound_unpool2 = Conv2DTranspose(256, 2, 2, name="boundary_deconv2")
        self.bound_unpool1 = Conv2DTranspose(128, 2, 2, name="boundary_deconv1")
        self.bound_unpool0 = Conv2DTranspose(64, 2, 2, name="boundary_deconv0")

        # self.bound_conv1 = Conv2D(
        #     64, 1, padding="same", activation=None, name="boundary_conv"
        # )
        self.bound_conv2 = Conv2D(
            2,
            1,
            padding="same",
            name="boundary_probability_map",
            activation=tf.keras.activations.softmax,
        )
        self.bound_mask = Lambda(
            lambda x: tf.expand_dims(tf.math.argmax(x, -1), -1),
            name="boundary_boundary_mask",
            trainable=False,
            dtype=tf.float32,
        )

        # ------------------------------------------------------------------------------
        # Boundary guided fusion decoder branch
        # ------------------------------------------------------------------------------
        self.fusion_bagm3 = BAGMBlock(in_channel=256, level=3, name="fusion_bagm3")
        self.fusion_bagm2 = BAGMBlock(in_channel=128, level=2, name="fusion_bagm2")
        self.fusion_bagm1 = BAGMBlock(in_channel=64, level=1, name="fusion_bagm1")
        self.fusion_bagm0 = BAGMBlock(in_channel=32, level=0, name="fusion_bagm0")

        if self.plane == "sagittal":
            self.fusion_resizing = Resizing(39, 32, name="fusion_resizing")
        elif self.plane == "coronal":
            self.fusion_resizing = Lambda(lambda x: x, name="fusion_resizing")
        else:
            self.fusion_resizing = Resizing(32, 39, name="fusion_resizing")

        # self.fusion_conv1 = Conv2D(
        #     64, 1, padding="same", activation=None, name="fusion_conv"
        # )

        self.fusion_conv2 = Conv2D(
            NUM_CLASS,
            1,
            padding="same",
            name="fusion_probability_map",
            activation=tf.keras.activations.softmax,
        )
        self.fusion_boundary_mask = BoundaryBlock(name="fusion_boundary_mask")
        self.fusion_segmentation_mask = Lambda(
            lambda x: tf.expand_dims(tf.math.argmax(x, -1), -1),
            name="fusion_segmentation_mask",
            trainable=False,
            dtype=tf.float32,
        )

        # ------------------------------------------------------------------------------
        # output branch
        # ------------------------------------------------------------------------------
        self.mean_segmentation_map = Average(name="mean_segmentation_map")
        self.final_conv2 = Conv2D(
            NUM_CLASS,
            1,
            padding="same",
            name="final_segmentation_map",
            activation=tf.keras.activations.softmax,
        )

        self.final_segmentation_mask = Lambda(
            lambda x: tf.expand_dims(tf.math.argmax(x, -1), -1),
            name="final_segmentation_mask",
            trainable=False,
            dtype=tf.float32,
        )

        self.final_boundary_mask = Average(name="final_boundary_mask")

    def call(self, inputs):

        inp1, inp2 = inputs

        # ----------------------------------------------------------------
        # encoder blocks
        # ----------------------------------------------------------------

        encoder0_out = self.encoder0(inp1)
        pool0_out = self.enc_pool0(encoder0_out["encoder_0_output"])

        encoder1_out = self.encoder1(pool0_out)
        pool1_out = self.enc_pool1(encoder1_out["encoder_1_output"])

        encoder2_out = self.encoder2(pool1_out)
        pool2_out = self.enc_pool2(encoder2_out["encoder_2_output"])

        encoder3_out = self.encoder3(pool2_out)
        encoder3_padding_out = self.enc_padding(encoder3_out["encoder_3_output"])
        pool3_out = self.enc_pool3(encoder3_padding_out)

        # ----------------------------------------------------------------
        # bottleneck
        # ----------------------------------------------------------------

        bottleneck_out = self.bottleneck(pool3_out)
        seg_unpool3_out = self.seg_unpool3(bottleneck_out["bottleneck_output"])

        # ----------------------------------------------------------------
        # skip connections
        # ----------------------------------------------------------------

        skip_con01_out = self.skip_conv01(encoder0_out["encoder_0_output"])
        skip_con12_out = self.skip_conv12(encoder1_out["encoder_1_output"])
        skip_con23_out = self.skip_conv23(encoder2_out["encoder_2_output"])

        skip_pooling01_out = self.skip_pooling01(skip_con01_out)
        skip_pooling12_out = self.skip_pooling12(skip_con12_out)
        skip_pooling23_out = self.skip_pooling23(skip_con23_out)
        skip_padding23_out = self.skip_padding(skip_pooling23_out)

        skip_add01_out = self.skip_add01(
            [encoder1_out["encoder_1_output"], skip_pooling01_out]
        )
        skip_add12_out = self.skip_add12(
            [encoder2_out["encoder_2_output"], skip_pooling12_out]
        )
        skip_add23_out = self.skip_add23([encoder3_padding_out, skip_padding23_out])

        # ----------------------------------------------------------------
        # Laplace blocks
        # ----------------------------------------------------------------

        lap_encoder0_out = self.lap_encoder0(inp2)
        lap_pool0_out = self.lap_enc_pool0(lap_encoder0_out["encoder_0_output"])
        lap_conv0_out = self.lap_conv0(lap_encoder0_out["encoder_0_output"])

        lap_encoder1_out = self.lap_encoder1(lap_pool0_out)
        lap_pool1_out = self.lap_enc_pool1(lap_encoder1_out["encoder_1_output"])
        lap_conv1_out = self.lap_conv1(lap_encoder1_out["encoder_1_output"])

        lap_encoder2_out = self.lap_encoder2(lap_pool1_out)
        lap_pool2_out = self.lap_enc_pool2(lap_encoder2_out["encoder_2_output"])
        lap_conv2_out = self.lap_conv2(lap_encoder2_out["encoder_2_output"])

        lap_encoder3_out = self.lap_encoder3(lap_pool2_out)
        lap_conv3_out = self.lap_conv3(lap_encoder3_out["encoder_3_output"])
        lap_conv3_out_resized = self.lap_enc_padding(lap_conv3_out)

        # ----------------------------------------------------------------
        # boundary decoder blocks
        # ----------------------------------------------------------------

        bound_unpool3_out = self.bound_unpool3(bottleneck_out["bottleneck_output"])
        bound_mult3 = self.bound_mult3([bound_unpool3_out, lap_conv3_out_resized])
        bound_decoder3_out = self.bound_decoder3([bound_mult3, skip_add23_out])
        bound_resizing = self.bound_resizing(bound_decoder3_out["decoder_3_output"])

        bound_unpool2_out = self.bound_unpool2(bound_resizing)
        bound_mult2 = self.bound_mult2([bound_unpool2_out, lap_conv2_out])
        bound_decoder2_out = self.bound_decoder2([bound_mult2, skip_add12_out])

        bound_unpool1_out = self.bound_unpool1(bound_decoder2_out["decoder_2_output"])
        bound_mult1 = self.bound_mult1([bound_unpool1_out, lap_conv1_out])
        bound_decoder1_out = self.bound_decoder1([bound_mult1, skip_add01_out])

        bound_unpool0_out = self.bound_unpool0(bound_decoder1_out["decoder_1_output"])
        bound_mult0 = self.bound_mult0([bound_unpool0_out, lap_conv0_out])
        bound_decoder0_out = self.bound_decoder0(
            [bound_mult0, encoder0_out["encoder_0_output"]]
        )

        # bound_conv1_out = self.bound_conv1(bound_decoder0_out["decoder_0_output"])
        bound_probability_map = self.bound_conv2(bound_decoder0_out["decoder_0_output"])
        bound_boundary_mask = tf.cast(
            self.bound_mask(bound_probability_map), tf.float32
        )

        # ----------------------------------------------------------------
        # segmentation decoder blocks
        # ----------------------------------------------------------------

        attn3_out = self.attn3([bottleneck_out["bottleneck_output"], skip_add23_out])
        seg_decoder3_out = self.seg_decoder3([seg_unpool3_out, attn3_out])
        seg_resizing_out = self.seg_resizing(seg_decoder3_out["decoder_3_output"])
        seg_unpool2_out = self.seg_unpool2(seg_resizing_out)

        attn2_out = self.attn2([seg_resizing_out, skip_add12_out])
        seg_decoder2_out = self.seg_decoder2([seg_unpool2_out, attn2_out])
        seg_unpool1_out = self.seg_unpool1(seg_decoder2_out["decoder_2_output"])

        attn1_out = self.attn1([seg_decoder2_out["decoder_2_output"], skip_add01_out])
        seg_decoder1_out = self.seg_decoder1([seg_unpool1_out, attn1_out])
        seg_unpool0_out = self.seg_unpool0(seg_decoder1_out["decoder_1_output"])

        attn0_out = self.attn0(
            [
                seg_decoder1_out["decoder_1_output"],
                encoder0_out["encoder_0_output"],
            ]
        )
        seg_decoder0_out = self.seg_decoder0([seg_unpool0_out, attn0_out])
        # seg_conv1_out = self.seg_conv1(seg_decoder0_out["decoder_0_output"])

        seg_probability_map = self.seg_conv2(seg_decoder0_out["decoder_0_output"])
        seg_boundary_mask = self.seg_boundary_mask(seg_probability_map)
        seg_segmentation_mask = self.seg_segmentation_mask(seg_probability_map)

        # ----------------------------------------------------------------
        # segmentation and boundary decoder fusion blocks
        # ----------------------------------------------------------------

        fusion_bagm3 = self.fusion_bagm3(
            [
                seg_decoder3_out["decoder_3_output"],
                bottleneck_out["bottleneck_output"],
                bound_decoder3_out["decoder_3_output"],
            ]
        )
        fusion_resizing_out = self.fusion_resizing(fusion_bagm3["bagm_3_output"])
        fusion_bagm2 = self.fusion_bagm2(
            [
                seg_decoder2_out["decoder_2_output"],
                fusion_resizing_out,
                bound_decoder2_out["decoder_2_output"],
            ]
        )
        fusion_bagm1 = self.fusion_bagm1(
            [
                seg_decoder1_out["decoder_1_output"],
                fusion_bagm2["bagm_2_output"],
                bound_decoder1_out["decoder_1_output"],
            ]
        )
        fusion_bagm0 = self.fusion_bagm0(
            [
                seg_decoder0_out["decoder_0_output"],
                fusion_bagm1["bagm_1_output"],
                bound_decoder0_out["decoder_0_output"],
            ]
        )
        # fusion_conv1_out = self.fusion_conv1(fusion_bagm0["bagm_0_output"])
        fusion_probability_map = self.fusion_conv2(fusion_bagm0["bagm_0_output"])
        fusion_boundary_mask = self.fusion_boundary_mask(fusion_probability_map)
        fusion_segmentation_mask = self.fusion_segmentation_mask(fusion_probability_map)

        # ----------------------------------------------------------------
        # output blocks
        # ----------------------------------------------------------------

        mean_segmentation_map_output = self.mean_segmentation_map(
            [fusion_probability_map, seg_probability_map]
        )
        final_segmentation_map_output = self.final_conv2(mean_segmentation_map_output)

        final_segmentation_mask_output = self.final_segmentation_mask(
            final_segmentation_map_output
        )

        final_boundary_mask_output = self.final_boundary_mask(
            [bound_boundary_mask, seg_boundary_mask, fusion_boundary_mask]
        )

        return {
            # final branch outputs
            "final_segmentation_map": final_segmentation_map_output,  # shape = (H, W, C)
            "final_segmentation_mask": final_segmentation_mask_output,  # shape = (H, W, 1)
            "final_boundary_mask": final_boundary_mask_output,  # shape = (H, W, 1)
            # segmentation branch outputs
            "seg_probability_map": seg_probability_map,  # shape = (H, W, C)
            "seg_boundary_mask": seg_boundary_mask,  # shape = (H, W, 1)
            "seg_segmentation_mask": seg_segmentation_mask,  # shape = (H, W, 1)
            # fusion branch outputs
            "fusion_probability_map": fusion_probability_map,  # shape = (H, W, C)
            "fusion_boundary_mask": fusion_boundary_mask,  # shape = (H, W, 1)
            "fusion_segmentation_mask": fusion_segmentation_mask,  # shape = (H, W, 1)
            # boundary branch outputs
            "boundary_mask": bound_boundary_mask,  # shape = (H, W, 1)
            "boundary_probability_map": bound_probability_map,  # shape = (H, W, 2)
        }

    @tf.function
    def train_step(self, data):

        (x_true, laplace_true, inv_map_true), (y_true, boundary_true) = data

        CE_loss_seg = Multiclass_Weighted_CE()
        CE_loss_bound = Multiclass_Weighted_CE(num_cls=2)
        PCE_loss_seg = Pixelwise_Weighted_CE(inv_map_true)

        F_loss_seg = Multiclass_Weighted_FocalLoss()
        F_loss_bound = Multiclass_Weighted_FocalLoss(num_cls=2)
        PF_loss_seg = Pixelwise_Weighted_FocalLoss(inv_map_true)

        d_loss_seg = Multi_Weighted_SoftMax_DiceLoss()
        d_loss_fusion = Multi_Weighted_SoftMax_DiceLoss()

        with tf.GradientTape() as tape:
            y_pred = self([x_true, laplace_true], training=True)
            # Forward pass
            # Compute our own loss

            ce_loss_seg = CE_loss_seg(y_true, y_pred["final_segmentation_map"])
            pce_loss_seg = PCE_loss_seg(y_true, y_pred["final_segmentation_map"])
            ce_loss_bound = CE_loss_bound(
                boundary_true, y_pred["boundary_probability_map"]
            )

            f_loss_seg = F_loss_seg(y_true, y_pred["final_segmentation_map"])
            pf_loss_seg = PF_loss_seg(y_true, y_pred["final_segmentation_map"])
            f_loss_bound = F_loss_bound(
                boundary_true, y_pred["boundary_probability_map"]
            )

            dice_loss_seg = d_loss_seg(y_true, y_pred["seg_probability_map"])
            dice_loss_fusion = d_loss_fusion(y_true, y_pred["fusion_probability_map"])

            wt1 = tf.stop_gradient(pf_loss_seg // dice_loss_seg)
            wt2 = tf.stop_gradient(pf_loss_seg // dice_loss_fusion)

            total_mean_loss = (
                ce_loss_seg
                + pce_loss_seg
                + ce_loss_bound
                + f_loss_seg
                + pf_loss_seg
                + f_loss_bound
                + wt1 * dice_loss_seg
                + wt2 * dice_loss_fusion
            ) / 8

            mean_loss = (
                pf_loss_seg
                + f_loss_bound
                + wt1 * dice_loss_seg
                + wt2 * dice_loss_fusion
            ) / 4

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(mean_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        seg_d_sim1.update_state(y_true, y_pred["seg_segmentation_mask"])
        seg_d_sim2.update_state(y_true, y_pred["seg_segmentation_mask"])
        seg_d_sim3.update_state(y_true, y_pred["seg_segmentation_mask"])

        fusion_d_sim1.update_state(y_true, y_pred["fusion_segmentation_mask"])
        fusion_d_sim2.update_state(y_true, y_pred["fusion_segmentation_mask"])
        fusion_d_sim3.update_state(y_true, y_pred["fusion_segmentation_mask"])

        final_d_sim1.update_state(y_true, y_pred["final_segmentation_mask"])
        final_d_sim2.update_state(y_true, y_pred["final_segmentation_mask"])
        final_d_sim3.update_state(y_true, y_pred["final_segmentation_mask"])

        bound_d_sim1.update_state(boundary_true, y_pred["final_boundary_mask"])
        bound_d_sim2.update_state(boundary_true, y_pred["final_boundary_mask"])
        bound_d_sim3.update_state(boundary_true, y_pred["final_boundary_mask"])

        # return f"\nThe total loss is {mean_loss}\n dice loss for all classes is {dice_loss1}"

        return {
            "total_mean_loss": total_mean_loss,
            "mean_loss": mean_loss,
            # segmentation branch metrics
            "segmentation_dice_loss": dice_loss_seg,
            "segmentation_dice_similarity": seg_d_sim1.result(),
            "segmentation_selected_dice_similarity": seg_d_sim2.result(),
            "segmentation_avg_dice_similarity": seg_d_sim3.result(),
            # fusion branch metrics
            "fusion_dice_loss": dice_loss_fusion,
            "fusion_dice_similarity": fusion_d_sim1.result(),
            "fusion_selected_dice_similarity": fusion_d_sim2.result(),
            "fusion_avg_dice_similarity": fusion_d_sim3.result(),
            # output branch metrics
            "final_cross_entropy_loss": ce_loss_seg,
            "final_pixelwise_cross_entropy_loss": pce_loss_seg,
            "final_focal_loss": f_loss_seg,
            "final_pixelwise_focal_loss": pf_loss_seg,
            "final_segmentation_dice_similarity": final_d_sim1.result(),
            "final_segmentation_selected_dice_similarity": final_d_sim2.result(),
            "final_avg_dice_similarity": final_d_sim3.result(),
            "final_boundary_dice_similarity": bound_d_sim1.result(),
            "final_boundary_selected_dice_similarity": bound_d_sim2.result(),
            "final_boundary_avg_dice_similarity": bound_d_sim3.result(),
            # boundary branch metrics
            "boundary_cross_entropy_loss": ce_loss_bound,
            "boundary_focal_loss": f_loss_bound,
        }

    def test_step(self, data):

        (x_true, laplace_true, inv_map_true), (y_true, boundary_true) = data

        CE_loss_seg = Multiclass_Weighted_CE()
        CE_loss_bound = Multiclass_Weighted_CE(num_cls=2)
        PCE_loss_seg = Pixelwise_Weighted_CE(inv_map_true)

        F_loss_seg = Multiclass_Weighted_FocalLoss()
        F_loss_bound = Multiclass_Weighted_FocalLoss(num_cls=2)
        PF_loss_seg = Pixelwise_Weighted_FocalLoss(inv_map_true)

        d_loss_seg = Multi_Weighted_SoftMax_DiceLoss()
        d_loss_fusion = Multi_Weighted_SoftMax_DiceLoss()

        y_pred = self([x_true, laplace_true], training=False)  # Forward pass

        ce_loss_seg = CE_loss_seg(y_true, y_pred["final_segmentation_map"])
        pce_loss_seg = PCE_loss_seg(y_true, y_pred["final_segmentation_map"])
        ce_loss_bound = CE_loss_bound(boundary_true, y_pred["boundary_probability_map"])

        f_loss_seg = F_loss_seg(y_true, y_pred["final_segmentation_map"])
        pf_loss_seg = PF_loss_seg(y_true, y_pred["final_segmentation_map"])
        f_loss_bound = F_loss_bound(boundary_true, y_pred["boundary_probability_map"])

        dice_loss_seg = d_loss_seg(y_true, y_pred["seg_probability_map"])
        dice_loss_fusion = d_loss_fusion(y_true, y_pred["fusion_probability_map"])

        wt1 = tf.stop_gradient(pf_loss_seg // dice_loss_seg)
        wt2 = tf.stop_gradient(pf_loss_seg // dice_loss_fusion)

        total_mean_loss = (
            ce_loss_seg
            + pce_loss_seg
            + ce_loss_bound
            + f_loss_seg
            + pf_loss_seg
            + f_loss_bound
            + wt1 * dice_loss_seg
            + wt2 * dice_loss_fusion
        ) / 8

        mean_loss = (
            pf_loss_seg + f_loss_bound + wt1 * dice_loss_seg + wt2 * dice_loss_fusion
        ) / 4

        seg_d_sim1.update_state(y_true, y_pred["seg_segmentation_mask"])
        seg_d_sim2.update_state(y_true, y_pred["seg_segmentation_mask"])
        seg_d_sim3.update_state(y_true, y_pred["seg_segmentation_mask"])

        fusion_d_sim1.update_state(y_true, y_pred["fusion_segmentation_mask"])
        fusion_d_sim2.update_state(y_true, y_pred["fusion_segmentation_mask"])
        fusion_d_sim3.update_state(y_true, y_pred["fusion_segmentation_mask"])

        final_d_sim1.update_state(y_true, y_pred["final_segmentation_mask"])
        final_d_sim2.update_state(y_true, y_pred["final_segmentation_mask"])
        final_d_sim3.update_state(y_true, y_pred["final_segmentation_mask"])

        bound_d_sim1.update_state(boundary_true, y_pred["final_boundary_mask"])
        bound_d_sim2.update_state(boundary_true, y_pred["final_boundary_mask"])
        bound_d_sim3.update_state(boundary_true, y_pred["final_boundary_mask"])

        return {
            "total_mean_loss": total_mean_loss,
            "mean_loss": mean_loss,
            # segmentation branch metrics
            "segmentation_dice_loss": dice_loss_seg,
            "segmentation_dice_similarity": seg_d_sim1.result(),
            "segmentation_selected_dice_similarity": seg_d_sim2.result(),
            "segmentation_avg_dice_similarity": seg_d_sim3.result(),
            # fusion branch metrics
            "fusion_dice_loss": dice_loss_fusion,
            "fusion_dice_similarity": fusion_d_sim1.result(),
            "fusion_selected_dice_similarity": fusion_d_sim2.result(),
            "fusion_avg_dice_similarity": fusion_d_sim3.result(),
            # output branch metrics
            "final_cross_entropy_loss": ce_loss_seg,
            "final_pixelwise_cross_entropy_loss": pce_loss_seg,
            "final_focal_loss": f_loss_seg,
            "final_pixelwise_focal_loss": pf_loss_seg,
            "final_segmentation_dice_similarity": final_d_sim1.result(),
            "final_segmentation_selected_dice_similarity": final_d_sim2.result(),
            "final_avg_dice_similarity": final_d_sim3.result(),
            "final_boundary_dice_similarity": bound_d_sim1.result(),
            "final_boundary_selected_dice_similarity": bound_d_sim2.result(),
            "final_boundary_avg_dice_similarity": bound_d_sim3.result(),
            # boundary branch metrics
            "boundary_cross_entropy_loss": ce_loss_bound,
            "boundary_focal_loss": f_loss_bound,
        }

    def summary(self, **kwargs):
        inp1 = Input(shape=INPUT_SHAPE[self.plane])
        inp2 = Input(shape=INPUT_SHAPE[self.plane])

        model = Model(
            inputs=[inp1, inp2],
            outputs=self.call([inp1, inp2]),
            name="ResConvNet Model_(" + Path(MODEL_DIR).parent.stem + ")",
        )
        return model.summary(**kwargs)

    def plot_model_(self, **kwargs):
        inp1 = Input(shape=INPUT_SHAPE[self.plane])
        inp2 = Input(shape=INPUT_SHAPE[self.plane])
        model = Model(
            inputs=[inp1, inp2],
            outputs=self.call([inp1, inp2]),
            name="ResConvNet Model_(" + Path(MODEL_DIR).parent.stem + ")",
        )
        plot_model(model, **kwargs)

    def build_(self):
        inp1 = Input(shape=tuple(INPUT_SHAPE[self.plane]))
        inp2 = Input(shape=tuple(INPUT_SHAPE[self.plane]))

        model = Model(inputs=[inp1, inp2], outputs=self([inp1, inp2]))
        # return model


class ResidualDenseConvNet(BaseModel):
    def __init__(
        self,
        plane="sagittal",
        num_cls=NUM_CLASS,
        epsilon=EPSILON,
    ):
        assert plane in PLANE_IDX.keys()

        self.plane = plane
        self.num_cls = num_cls
        self.epsilon = epsilon
        self.input_shape = [tuple(INPUT_SHAPE[self.plane])] * 2
        self.model = ResDenseConvNet(
            name="DenseConvNet_" + self.plane,
            pooling="spp",
            atrous=False,
            plane=self.plane,
            skip_connection="concat",
        )
        # inp1 = Input(shape=tuple(INPUT_SHAPE[self.plane]))
        # inp2 = Input(shape=tuple(INPUT_SHAPE[self.plane]))
        # print(inp1.shape)
        self.model.build_()
        # self.model([inp1, inp2])
        self.model.compile(optimizer=optimizers.Adam())

        # for i, w in enumerate(self.model.weights):
        #     print(i, w.name)

    def plot_model(self, **kwargs):
        model = self.get_model_()
        model.plot_model_(**kwargs)

    def load_generator_(
        self,
        directory,
        folder,
        batch_size=BATCH_SIZE,
        orientation="LAS",
    ):
        datagen = DataLoader(
            directory, folder, batch_size, self.plane, orientation
        ).get_generator_()
        return datagen
