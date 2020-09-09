# coding: utf-8

"""
@File   : sampler.py
@Author : garnet
@Time   : 2020/9/9 17:56
"""

from .dataset import Dataset, MappingDataset


class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements.
    """

    def __init__(self, dataset: Dataset, *args, **kwargs):
        pass

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    """

    def __init__(self, dataset: MappingDataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
