# coding: utf-8

"""
@File   : normalization.py
@Author : garnet
@Time   : 2020/10/30 11:53
"""

from functools import reduce
from scipy.stats import truncnorm


def truncated_normal(shape, mean=0., stddev=0.02, lower=2, upper=2):
    total = shape if isinstance(shape, int) else reduce(lambda x, y: x * y, shape, 1)
    tn = truncnorm(-abs(lower), upper, loc=mean, scale=stddev)
    samples = tn.rvs(total).reshape((shape,) if isinstance(shape, int) else shape)
    return samples
