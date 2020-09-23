# coding: utf-8

"""
@File   : fetcher.py
@Author : garnet
@Time   : 2020/9/10 17:01
"""


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
            data = [self.dataset[batch_indices]]
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
            data = [next(self.dataset)]
        return self.collator.collate_fn(data)
