# coding: utf-8

"""
@File   : collator.py
@Author : garnet
@Time   : 2020/9/10 16:20
"""

import six
import typing
import numpy as np


class Collator(object):
    def collate_fn(self, batch_data):
        raise NotImplementedError


class IdenticalCollator(Collator):
    def collate_fn(self, batch_data):
        return batch_data


class SingleSampleCollator(Collator):
    r"""Convert single sample data into keras supported format.

    Inputs for model training must be a single array or a list or arrays.
    """

    def collate_fn(self, batch_data):
        if isinstance(batch_data, np.ndarray):
            return batch_data
        elif isinstance(batch_data, tuple):
            return tuple(*(self.collate_fn(d) for d in batch_data))
        elif isinstance(batch_data, typing.Mapping):
            return {key: self.collate_fn(batch_data[key]) for key in batch_data}
        elif isinstance(batch_data, typing.Sequence) and not isinstance(batch_data, six.string_types):
            return [np.array(d) for d in batch_data]
        else:
            return batch_data


class BatchSampleCollator(Collator):
    r"""Puts each data field into a tensor with outer dimension batch size

    Inputs for model training must be a single array or a list or arrays.
    """

    def collate_fn(self, batch_data):
        sample = batch_data[0]
        if isinstance(sample, np.ndarray):
            return np.stack(batch_data, axis=0)
        elif isinstance(sample, float):
            return np.array(batch_data, dtype=np.float32)
        elif isinstance(sample, six.integer_types):
            return np.array(batch_data, dtype=np.int32)
        elif isinstance(sample, six.string_types):
            return batch_data
        elif isinstance(sample, typing.Mapping):
            return {key: self.collate_fn([d[key] for d in batch_data]) for key in sample}
        elif isinstance(sample, tuple):
            return tuple([self.collate_fn(piece) for piece in zip(*batch_data)])
        elif isinstance(sample, typing.Sequence):
            return [np.array(piece) for piece in zip(*batch_data)]
        raise TypeError("batch must contain numpy arrays, numbers, dicts or lists; found {}".format(type(sample)))
