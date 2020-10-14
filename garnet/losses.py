# coding: utf-8

"""
@File   : losses.py
@Author : garnet
@Time   : 2020/10/14 17:05
"""

import keras
import tensorflow as tf
import keras.backend as K
import numpy as np


def gelu_erf(x):
    r"""Approximate formula of Gaussian Error Linear Units (GELUs) loss function, which based on erf.
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    r"""Approximate formula of Gaussian Error Linear Units (GELUs) loss function, which based on tanh.
    """
    cdf = 0.5 * (1.0 + K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


def set_gelu(version='erf'):
    r"""Set the version of gelu loss function.
    """
    version = version.lower()
    assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
    if version == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh


custom_objects = {
    'gelu_erf': gelu_erf,
    'gelu_tanh': gelu_tanh,
    'gelu': gelu_erf,
}

keras.utils.get_custom_objects().update(custom_objects)
