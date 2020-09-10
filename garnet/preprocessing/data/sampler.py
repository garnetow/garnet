# coding: utf-8

"""
@File   : sampler.py
@Author : garnet
@Time   : 2020/9/9 17:56
"""

import random

from .dataset import Dataset, MappingDataset


class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements.
    """

    def __init__(self, dataset: Dataset = None, *args, **kwargs):
        pass

    def __iter__(self):
        raise NotImplementedError


class MappingSampler(Sampler):
    def __len__(self):
        raise NotImplementedError


class InfiniteStreamSampler(Sampler):
    r"""Used as sampler for :class:`IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None


class SequentialSampler(MappingSampler):
    r"""Samples elements sequentially, always in the same order.
    """

    def __init__(self, dataset: MappingDataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class RandomSampler(MappingSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    """

    def __init__(self, dataset: MappingDataset, num_samples=None, bootstrap=False):
        super().__init__(dataset)
        self.dataset = dataset
        self._num_samples = num_samples
        self.bootstrap = bootstrap

    @property
    def num_samples(self):
        return len(self.dataset) if self._num_samples is None else self._num_samples

    def __iter__(self):
        total_indices = list(range(len(self.dataset)))
        if self.bootstrap:
            return iter(random.choices(total_indices, k=self.num_samples))
        else:
            return iter(random.sample(total_indices, k=self.num_samples))

    def __len__(self):
        return self.num_samples


class BatchSampler(MappingSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Base sampler must have :meth:`__len__`, which in other words, provides a way to iterate over indices of mapping
    dataset elements.

    Arguments:
        :param sampler (Sampler): Base sampler.
        :param batch_size (int): Size of mini-batch.
        :param drop_last (bool, default: False): If `True`, the sampler will drop the last batch if
            its size would be less than `batch_size`.

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last: bool = False):
        super().__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
