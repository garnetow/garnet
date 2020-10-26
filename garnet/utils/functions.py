# coding: utf-8

"""
@File   : functions.py
@Author : garnet
@Time   : 2020/10/26 14:40
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)
