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


class SecondSequenceCrossEntropy(LossLayer):
    def compute_loss(self, inputs, segments=None, mask=None):
        y_true, y_pred = inputs
        y_true = y_true[:, 1:]  # target token ids
        y_mask = segments[:, 1:]  # segment ids
        y_pred = y_pred[:, :-1]  # predict probabilities
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)  # `y_true` is not one-hot, use sparse method
        loss = K.sum(loss * y_mask) / K.sum(y_mask)  # only care about the prediction result of second segment
        return loss
