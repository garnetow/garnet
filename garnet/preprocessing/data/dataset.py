# coding: utf-8

"""
@File   : dataset.py
@Author : garnet
@Time   : 2020/9/9 14:48
"""


class Dataset(object):
    r"""An abstract class representing a :class:`Dataset`.

    Data should be stored in a :class:`Dataset`. There are two types of :class:`Dataset`
    - Mapping style: :class:`MappingDataset`
    - Iterable style: :class:`IterableDataset`
    """

    def __iter__(self):
        raise self

    def __next__(self):
        raise NotImplementedError


class MappingDataset(Dataset):
    r"""An abstract class representing a map from keys to data samples.

    Data should be pre-loaded and stored in a :class:`MappingDataset`, in other words, data are stored in memory when
    an instance of :class:`MappingDataset` has been created.

    Implement :meth:`__getitem__` and :meth:`__len__` for subclasses.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        self._index = iter(range(len(self)))
        return self

    def __next__(self):
        return self[next(self._index)]


class IterableDataset(Dataset):
    r"""An iterable Dataset abstract, class, representing an iterable of data samples.
    Such form of datasets is particularly useful when data come from a stream, e.g. file stream.

    Implement :meth:`iter` for subclasses.
    """

    def iter(self):
        raise NotImplementedError

    def __iter__(self):
        self._iterator = self.iter()
        return self

    def __next__(self):
        return next(self._iterator)


class MultiListDataset(MappingDataset):
    r"""Dataset wrapping list-like data.

    Each sample will be retrieved by indexing data along the first dimension.

    Arguments:
        :param tensors that have the same size of the first dimension.
    """
    def __init__(self, *lists):
        assert all(len(single) == len(single[0]) for single in lists)
        self.data = lists

    def __getitem__(self, index):
        return [single[index] for single in self.data]

    def __len__(self):
        return len(self.data[0])
