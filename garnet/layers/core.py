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
        self.initializer = keras.activations.get(initializer)
        self.bias = None

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias',
            shape=(output_dim,),
            initializer=self.initializer,
        )
        super(Bias, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return K.bias_add(inputs, bias=self.bias)