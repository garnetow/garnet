# coding: utf-8

"""
@File   : dataset.py
@Author : garnet
@Time   : 2020/8/12 9:17
"""

import tensorflow as tf


class Dataset(object):
    """
    Data should be stored in a :class:`Dataset`. There are two types of :class:`Dataset`
    - Mapping style
    - Iterable style
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class MappingDataset(Dataset):
    """
    Represent a map from keys to data samples. Data should be pre-loaded and stored in a :class:`MappingDataset`,
    in other words, data are stored in memory when an instance of :class:`MappingDataset` has been created.
    """

    def __len__(self):
        raise NotImplementedError


class IterableDataset(Dataset):
    """
    An iterable Dataset, representing an iterable of data samples. Such form of datasets is particularly useful when
    data come from a stream, e.g. file stream.

    :meth:`__len__` is not provided in :class:`IterableDataset`.
    """

from keras.models import Model

model = Model().fit_generator

class TFRecordDataset(IterableDataset):
    def __init__(self, record_name):
        self.record = tf.data.TFRecordDataset(record_name)
        self.record.prefetch()
        self.record.map()

    def __iter__(self):
        return self.record.__iter__()
