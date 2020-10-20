# coding: utf-8

"""
@File   : similarity.py
@Author : garnet
@Time   : 2020/10/20 23:37
"""

import numpy as np

from ..distance import wasserstein_distance


def word_mover_distance(x, y):
    """
    WMD(Word Mover's Distance).

    Input shape:
    x.shape=[m,d], y.shape=[n,d], `m` and `n` are sequence lengths and `d` is hidden size
    """
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    D = np.sqrt(np.square(x[:, None] - y[None, :]).mean(axis=2))
    return wasserstein_distance(p, q, D)


def word_rotator_distance(x, y):
    """
    WRD(Word Rotator's Distance)

    Input shape:
    x.shape=[m,d], y.shape=[n,d], `m` and `n` are sequence lengths and `d` is hidden size
    """
    x_norm = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_norm = (y ** 2).sum(axis=1, keepdims=True) ** 0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    return wasserstein_distance(p, q, D)


def word_rotator_similarity(x, y):
    """
    1 - WRD
    """
    return 1 - word_rotator_distance(x, y)
