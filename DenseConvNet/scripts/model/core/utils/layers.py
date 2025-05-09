import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Add,
    Input,
    MaxPooling2D,
    AveragePooling2D,
    Multiply,
)


class MixedPooling2D(Layer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        alpha=0.5,
        **kwargs,
    ):

        super(MixedPooling2D, self).__init__(**kwargs)
        self.mean_pool = AveragePooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        self.max_pool = MaxPooling2D(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        self.add = Add(name="mixedpooling_output")
        self.alpha = alpha

    def call(self, inputs):

        mean_pool = self.mean_pool(inputs)
        max_pool = self.max_pool(inputs)

        max_p = self.alpha * max_pool
        mean_p = (1 - self.alpha) * mean_pool

        return self.add([max_p, mean_p])

    def summary(self, input_shape=(256, 256, 1), **kwargs):
        inputs = Input(shape=input_shape)
        m = Model(inputs=inputs, outputs=self.call(inputs), name="mixedpooling_layer")
        return m.summary(**kwargs)
