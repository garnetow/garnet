# coding: utf-8

"""
@File   : collator.py
@Author : garnet
@Time   : 2020/9/10 16:20
"""

import numpy as np


class Collator(object):
    def collate_fn(self, batch_data):
        raise NotImplementedError


class SampleConvertCollator(Collator):
    r"""Convert data into keras supported format.

    Inputs for model training must be a single array or a list or arrays.
    """
    def collate_fn(self, batch_data):
        if isinstance(batch_data, np.ndarray):
            return batch_data
        elif isinstance(batch_data, tuple):
            return tuple(*(self.collate_fn(d) for d in batch_data))

