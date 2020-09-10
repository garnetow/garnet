# coding: utf-8

"""
@File   : dataloader.py
@Author : garnet
@Time   : 2020/9/10 10:28
"""

from .dataset import Dataset, DatasetKind
from .sampler import Sampler, BatchSampler, InfiniteStreamSampler, SequentialSampler, RandomSampler
from .collator import Collator


class DataLoader(object):
    r"""Data loader, Combines a dataset and a sampler, and provides an iterable over the given dataset.

    :class:`DataLoader` supports both map-style and iterable-style datasets with single or multi-process loading,
    customizing loading order and optional automatic batching.

    Arguments:
        :param dataset (:class:`Dataset`): dataset from which to load the data.
        :param batch_size (int, optional, default: 1): how many samples per batch to load.
        :param shuffle (bool, optional, default: False): set to `True` to have the data reshuffled at every epoch.
        :param drop_last (bool, optional, default: False): set to `True` to drop the last incomplete batch.
        :param sampler (:class:`Sampler`, optional, default: None): defines the strategy to draw samples
            from the dataset.
        :param batch_sampler (:class:`BatchSampler`, optional, default: None): returns a batch of indices at a time.
            Mutually exclusive with :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`.
        :param collator (:class:`Collator`, optional, default: None): merges a list of samples to form a
            mini-batch. Used when using batched loading from a map-style dataset.
        :param num_workers (int, optional, default: 0): how many subprocesses to use for data loading. `0` means that
            the data will be loaded in the main process.
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 sampler: Sampler = None,
                 batch_sampler: BatchSampler = None,
                 collator: Collator = None,
                 num_workers: int = 0):
        self.dataset = dataset
        self._dataset_kind = self.dataset.kind
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset.kind == DatasetKind.Iterable:
            if sampler is not None:
                raise ValueError("DataLoader with IterableDataset: expected unspecified sampler option, "
                                 "but got sampler={}".format(sampler))
            if batch_sampler is not None:
                raise ValueError("DataLoader with IterableDataset: expected unspecified batch_sampler option, "
                                 "but got batch_sampler={}".format(batch_sampler))

        # check compatibility of :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, :attr:`batch_sampler`
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError("batch_sampler option is mutually exclusive with "
                                 "batch_size, shuffle, sampler, and drop_last")
            batch_size = None
            drop_last = False
        elif batch_size is None:
            if drop_last:
                raise ValueError("batch_size=None option disables auto-batching "
                                 "and is mutually exclusive with drop_last")

        if sampler is None:  # auto-generating default sampler
            if dataset.kind == DatasetKind.Iterable:
                sampler = InfiniteStreamSampler()
            else:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_sampler is None and batch_size is not None:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if collator is None:
            pass  # TODO: Collator class
        self.collator = collator

    def __iter__(self):
        if self.num_workers == 0:
            pass
        else:
            pass

    @property
    def auto_batch(self):
        return self.batch_sampler is not None

    @property
    def index_sampler(self):
        return self.batch_sampler if self.auto_batch else self.sampler


class BaseDataIterator(object):
    def __init__(self, data_loader: DataLoader):
        self.dataset = data_loader.dataset
        self._dataset_kind = data_loader._dataset_kind
        self.shuffle = data_loader.shuffle
        self.drop_last = data_loader.drop_last
        self.batch_size = data_loader.batch_size
        self.sampler = data_loader.sampler
        self.batch_sampler = data_loader.batch_sampler
        self.index_sampler = data_loader.index_sampler
        self._index_iter = iter(self.index_sampler)
        self.num_workers = data_loader.num_workers
        self.collator = data_loader.collator

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._index_iter)

    def _next_data(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class MappingDataIterator(BaseDataIterator):
    def __init__(self, data_loader):
        super().__init__(data_loader)
        self._num_yielded = 0

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self):
        return len(self.index_sampler)


class SingleProcessMappingDataIterator(MappingDataIterator):
    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.fetcher = None  # TODO: Fetcher
