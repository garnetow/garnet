# coding: utf-8

"""
@File   : fetcher.py
@Author : garnet
@Time   : 2020/9/10 17:01
"""

import random
from enum import Enum


class Fetcher(object):
    def __init__(self, dataset, auto_batch, collator, drop_last):
        self.dataset = dataset
        self._auto_batch = auto_batch
        self.collator = collator
        self._drop_last = drop_last

    def fetch(self, batch_indices):
        raise NotImplementedError


class MappingFetcher(Fetcher):
    def fetch(self, batch_indices):
        if self._auto_batch:
            data = [self.dataset[index] for index in batch_indices]
        else:
            data = self.dataset[batch_indices]
        return self.collator.collate_fn(data)


class IterableFetcher(Fetcher):
    def fetch(self, batch_indices):
        if self._auto_batch:
            data = []
            for _ in batch_indices:
                try:
                    data.append(next(self.dataset))
                except StopIteration:
                    break
            if len(data) == 0 or (self._drop_last and len(data) < len(batch_indices)):
                raise StopIteration
        else:
            data = next(self.dataset)
        return self.collator.collate_fn(data)


class _BufferStatus(Enum):
    Initial = 0
    Reading = 1
    Pending = 2


class IterableBufferFetcher(Fetcher):
    def __init__(self, dataset, auto_batch, collator, drop_last, buffer_size=1024):
        super().__init__(dataset, auto_batch, collator, drop_last)
        self.buffer_size = buffer_size
        self.buffer = []
        self.status = _BufferStatus.Initial

    def fetch(self, batch_indices):
        if self.status == _BufferStatus.Initial or self.status == _BufferStatus.Reading:
            while len(self.buffer) < self.buffer_size:
                try:
                    self.buffer.append(next(self.dataset))
                    self.status = _BufferStatus.Reading
                except StopIteration:
                    self.status = _BufferStatus.Pending
                    break

        if self._auto_batch:
            data = []
            for _ in batch_indices:
                if self.buffer:
                    index = random.randrange(len(self.buffer))
                    data.append(self.buffer.pop(index))
                else:
                    break
        else:
            index = random.randrange(len(self.buffer))
            data = self.buffer.pop(index)

        if len(self.buffer) == 0:
            self.status = _BufferStatus.Initial

        return self.collator.collate_fn(data)
