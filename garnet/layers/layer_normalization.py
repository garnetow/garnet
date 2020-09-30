# coding: utf-8

"""
@File   : layer_normalization.py
@Author : garnet
@Time   : 2020/9/30 23:55
"""

import keras
import keras.backend as K
from keras.layers import Layer


class LayerNormalization(Layer):
    r"""Layer normalization with conditions.

    See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

    Argument:
        :param center (:obj:`bool`, optional, default: `True`):
            Add an offset parameter if it is True.
    """

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 conditional=False,
                 hidden_units=None,
                 hidden_activation='linear',
                 hidden_initializer='glorot_uniform',
                 **kwargs):
        super().__init__(**kwargs)
