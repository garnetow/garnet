# coding: utf-8

"""
@File   : core.py
@Author : garnet
@Time   : 2020/10/13 19:05
"""

import keras
import keras.backend as K
from keras.layers import Layer


class BiasAdd(Layer):
    r"""Add trainable bias to the input
    """

    def __init__(self, initializer='zeros', **kwargs):
        super(BiasAdd, self).__init__(**kwargs)
        self.supports_masking = True

        self.initializer = keras.initializers.get(initializer)
        self.bias = None

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias',
            shape=(output_dim,),
            initializer=self.initializer,
        )
        super(BiasAdd, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.bias_add(inputs, bias=self.bias)


custom_objects = {
    'BiasAdd': BiasAdd,
}

keras.utils.get_custom_objects().update(custom_objects)
