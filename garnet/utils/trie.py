# coding: utf-8

"""
@File   : trie.py
@Author : garnet
@Time   : 2020/12/25 11:38
"""

import datrie
import string
from functools import wraps


def unicode2latin(unicode_string):
    return unicode_string.encode("utf-8").decode("latin1")


def latin2unicode(latin_string):
    return latin_string.encode("latin1").decode("utf-8")


class DoubleArrayTrie(object):
    @classmethod
    def unicode_adapter_factory(cls):
        class UnicodeAdapter(object):
            def __init__(self, _class):
                self._class = _class

            def __call__(self):
                pass

        return UnicodeAdapter

    def __init__(self, alphabet=None, ranges=None):
        """
        Trie needs to know the range of unicode symbols for efficiency, either `alphabet`  or `ranges` must be applied.
        """
        if isinstance(alphabet, str):
            self.trie = datrie.Trie(alphabet)
        elif ranges is not None:
            self.trie = datrie.Trie(ranges=ranges)
        else:
            print("Either `alphabet` or `ranges` must be applied when initialing. Using english related chars")
            self.trie = datrie.Trie(string.printable)

    @staticmethod
    def encode_key(key):
        return key

    @staticmethod
    def decode_key(key):
        return key

    def unicode_input_adapt(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            new_args = list(args)
            if len(new_args) > 0:
                new_args[0] = self.encode_key(new_args[0])
            new_args = tuple(new_args)

            if "key" in kwargs:
                kwargs["key"] = self.encode_key(kwargs["key"])
            if "text" in kwargs:
                kwargs["text"] = self.encode_key(kwargs["text"])
            if "prefix" in kwargs:
                kwargs["prefix"] = self.encode_key(kwargs["prefix"])

            res = func(self, *new_args, **kwargs)
            return res

        return wrapper

    @unicode_input_adapt
    def __setitem__(self, key, value):
        self.trie[key] = value

    @unicode_input_adapt
    def __getitem__(self, key):
        return self.trie[key]

    @unicode_input_adapt
    def __delitem__(self, key):
        del self.trie[key]

    @unicode_input_adapt
    def prefixes(self, text):
        prefs = self.trie.prefixes(text)
        return [self.decode_key(p) for p in prefs]

    @unicode_input_adapt
    def prefix_items(self, text):
        items = self.trie.prefix_items(text)
        return [(self.decode_key(k), v) for k, v in items]

    @unicode_input_adapt
    def prefix_values(self, text):
        return self.trie.prefix_values(text)

    @unicode_input_adapt
    def iter_prefixes(self, text):
        for prefix in self.trie.iter_prefixes(text):
            yield self.decode_key(prefix)

    @unicode_input_adapt
    def iter_prefix_items(self, text):
        for k, v in self.trie.iter_prefix_items(text):
            yield self.decode_key(k), v

    @unicode_input_adapt
    def iter_prefix_values(self, text):
        for value in self.trie.iter_prefix_values(text):
            yield value

    @unicode_input_adapt
    def longest_prefix(self, text):
        return self.decode_key(self.trie.longest_prefix(text))

    @unicode_input_adapt
    def longest_prefix_item(self, text):
        item = self.trie.longest_prefix_item(text)
        return self.decode_key(item[0]), item[1] if isinstance(item, tuple) else item

    @unicode_input_adapt
    def longest_prefix_value(self, text):
        return self.trie.longest_prefix_value(text)

    @unicode_input_adapt
    def has_keys_with_prefix(self, prefix):
        return self.trie.has_keys_with_prefix(prefix)

    @unicode_input_adapt
    def keys_with_prefix(self, prefix):
        return [self.decode_key(t) for t in self.trie.keys(prefix)]

    @unicode_input_adapt
    def items_with_prefix(self, prefix):
        items = self.trie.items(prefix)
        return [(self.decode_key(k), v) for k, v in items] if items is not None else items

    @unicode_input_adapt
    def value_with_prefix(self, prefix):
        return self.trie.values(prefix)

    @unicode_input_adapt
    def suffixes(self, prefix=None):
        return [self.decode_key(suf) for suf in self.trie.suffixes(prefix)]

    def save(self, path):
        self.trie.save(path)

    @classmethod
    def load(cls, path):
        return datrie.Trie.load(path)


class ChineseTrie(DoubleArrayTrie):
    def __init__(self):
        super(ChineseTrie, self).__init__(ranges=[('\x01', '\xff')])

    @staticmethod
    def encode_key(key):
        return key.encode("utf-8").decode("latin1") if key is not None else key

    @staticmethod
    def decode_key(key):
        return key.encode("latin1").decode("utf-8") if key is not None else key
