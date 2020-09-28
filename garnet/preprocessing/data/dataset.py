# coding: utf-8

"""
@File   : dataset.py
@Author : garnet
@Time   : 2020/9/9 14:48
"""

import numpy as np
from enum import Enum


class DatasetKind(Enum):
    Unspecified = 0
    Map = 1
    Iterable = 2


class Dataset(object):
    r"""An abstract class representing a :class:`Dataset`.

    Data should be stored in a :class:`Dataset`. There are two types of :class:`Dataset`
    - Mapping style: :class:`MappingDataset`
    - Iterable style: :class:`IterableDataset`
    """
    kind = DatasetKind.Unspecified

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class MappingDataset(Dataset):
    r"""An abstract class representing a map from keys to data samples.

    Data should be pre-loaded and stored in a :class:`MappingDataset`, in other words, data are stored in memory when
    an instance of :class:`MappingDataset` has been created.

    Implement :meth:`__getitem__` and :meth:`__len__` for subclasses.
    """
    kind = DatasetKind.Map

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if getattr(self, '_internal_iter', None) is None:
            self._internal_iter = iter(range(len(self)))
        try:
            index = next(self._internal_iter)
            return self[index]
        except StopIteration:
            self._internal_iter = iter(range(len(self)))
            raise StopIteration


class IterableDataset(Dataset):
    r"""An iterable Dataset abstract, class, representing an iterable of data samples.
    Such form of datasets is particularly useful when data come from a stream, e.g. file stream.

    Implement :meth:`iter` for subclasses.
    """
    kind = DatasetKind.Iterable

    def iter(self):
        raise NotImplementedError

    def __next__(self):
        if getattr(self, '_internal_iter', None) is None:
            self._internal_iter = self.iter()
        try:
            data = next(self._internal_iter)
            return data
        except StopIteration:
            self._internal_iter = self.iter()
            raise StopIteration


class MatrixDataset(MappingDataset):
    r"""Dataset wrapping list-like data.

    Each sample will be retrieved by indexing data along the first dimension.

    Arguments:
        :param arrays: tensors that have the same size of the first dimension.
        :param labels: label matrices.
        :param y_out (bool, default: True): whether to export label. If `true`, a tuple in (x, y) format will be the
            result.
    """

    def __init__(self, arrays, labels=None, y_out=True):
        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        if labels is not None and isinstance(labels, np.ndarray):
            labels = [labels]
        assert all(len(single) == len(arrays[0]) for single in arrays)
        if labels is not None:
            assert all(len(single) == len(arrays[0]) for single in labels)
            assert y_out is True
        self.data = arrays
        self.labels = labels
        self._y_out = y_out

    def __getitem__(self, index):
        if self.has_label:
            return [single[index] for single in self.data], [single[index] for single in self.labels]
        else:
            if self._y_out:
                return [single[index] for single in self.data], None
            else:
                return [single[index] for single in self.data]

    def __len__(self):
        return len(self.data[0])

    @property
    def has_label(self):
        return True if self.labels else False
