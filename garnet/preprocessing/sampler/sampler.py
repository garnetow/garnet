# coding: utf-8

"""
@File   : sampler.py
@Author : garnet
@Time   : 2020/8/12 9:55
"""

from ..dataset.dataset import MappingDataset


class Sampler(object):
    """
    Sampler provides a way to iterate over specified dataset with :meth:`__iter__`.
    """

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, dataset: MappingDataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class InfiniteStreamSampler(Sampler):
    def __iter__(self):
        while True:
            yield None
