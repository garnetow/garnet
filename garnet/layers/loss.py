# coding: utf-8

"""
@File   : loss.py
@Author : garnet
@Time   : 2020/10/21 15:39
"""

import keras.backend as K
from keras.layers import Layer


class LossLayer(Layer):
    r"""Layer to define complex loss layer.
    """

    def __init__(self, **kwargs):
        super(LossLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        loss = self.compute_loss(inputs, mask=mask, **kwargs)
        self.add_loss(loss)  # add loss tensor into this layer, and then to the model
        return inputs

    def compute_loss(self, inputs, mask=None, **kwargs):
        raise NotImplementedError
