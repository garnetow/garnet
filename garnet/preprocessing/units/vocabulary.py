# coding: utf-8

"""
@File   : vocabulary.py
@Author : garnet
@Time   : 2020/10/15 8:21
"""

import json
import typing
import pathlib
from functools import wraps

from .base import StatefulUnit
from ...utils.io import safe_save, safe_save_json, check_assert_suffix
from ...utils.strings import is_cjk_character, is_punctuation_character

PAD = '[PAD]'
UNK = '[UNK]'
SOS = '[SOS]'  # start of sentence
EOS = '[EOS]'  # end of sentence
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'


class Vocabulary(StatefulUnit):
    def __init__(self,
                 token_start: typing.Union[bool, str] = False,
                 token_end: typing.Union[bool, str] = False,
                 token_pad: typing.Union[bool, str] = False,
                 token_unknown: typing.Union[bool, str] = False,
                 token_mask: typing.Union[bool, str] = False,
                 ignore_case=False,
                 **kwargs):
        super(Vocabulary, self).__init__(**kwargs)

        self.ignore_case = ignore_case

        self._token_start = token_start if isinstance(token_start, str) else SOS if token_start else None
        self._token_end = token_end if isinstance(token_end, str) else EOS if token_end else None
        self._token_pad = token_pad if isinstance(token_pad, str) else PAD if token_pad else None
        self._token_mask = token_mask if isinstance(token_mask, str) else MASK if token_mask else None
        self._token_unk = token_unknown if isinstance(token_unknown, str) else UNK if token_unknown else None
        self._special_tokens = None
        self.update_special_set()

        self._vocab = dict()
        self._vocab_reverse = dict()

    def update_special_set(self):
        self._special_tokens = set([token for token in [self._token_start, self._token_end, self._token_pad,
                                                        self._token_unk, self._token_mask] if token is not None])

    def lower(self, token: str):
        if token not in self._special_tokens:
            token = token.lower()
        return token

    def check_case(func):
        @wraps(func)
        def wrapper(self, token):
            token = self.lower(token) if self.ignore_case else token
            return func(self, token)

        return wrapper

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, mapping: dict):
        self._vocab = mapping
        self.update_reverse_vocab()
        self.fitted = True

    @property
    def start_token(self):
        return self._token_start

    @start_token.setter
    def start_token(self, token: str):
        self._token_start = token
        self.update_special_set()

    @property
    def end_token(self):
        return self._token_end

    @end_token.setter
    def end_token(self, token: str):
        self._token_end = token
        self.update_special_set()

    @property
    def pad_token(self):
        return self._token_pad

    @pad_token.setter
    def pad_token(self, token: str):
        self._token_pad = token
        self.update_special_set()

    @property
    def unknown_token(self):
        return self._token_unk

    @unknown_token.setter
    def unknown_token(self, token: str):
        self._token_unk = token
        self.update_special_set()

    @property
    def mask_token(self):
        return self._token_mask

    @mask_token.setter
    def mask_token(self, token: str):
        self._token_mask = token
        self.update_special_set()

    def update_reverse_vocab(self):
        vocab_rev = {v: k for k, v in self._vocab.items()}
        self._vocab_reverse = vocab_rev
        return vocab_rev

    @check_case
    def __getitem__(self, token):
        return self._vocab[token] if token in self._vocab else self._vocab.get(self._token_unk)

    @check_case
    def __contains__(self, token):
        return token in self._vocab

    @check_case
    def w2i(self, token):
        return self[token]

    def i2w(self, index):
        return self._vocab_reverse[index] if index in self._vocab_reverse else self._token_unk

    def transform(self, inputs: typing.Iterable, *args, **kwargs):
        if isinstance(inputs, str):
            return self[inputs]
        return [self[token] for token in inputs]

    def reverse_transform(self, inputs: typing.Union[typing.Iterable, int], *args, **kwargs):
        if isinstance(inputs, int):
            return self.i2w(inputs)
        return [self.i2w(index) for index in inputs]

    def init_with_tokens(self, tokens, special_tokens=None):
        vocab = dict()
        for st in [self._token_pad, self._token_start, self._token_end, self._token_unk, self._token_mask]:
            if st is not None:
                vocab[st] = len(vocab)

        if not isinstance(tokens, str):
            tokens = [self.lower(t) if self.ignore_case else t for t in tokens]
        else:
            tokens = [self.lower(t) if self.ignore_case else t for l in tokens for t in l]
        tokens = set(tokens)

        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        for token in special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        self.vocab = vocab

    def init_with_vocab_file(self, vocab_path, encoding='utf-8', special_tokens=None):
        path = pathlib.Path(vocab_path)
        with open(path, 'r', encoding=encoding) as f:
            vocab = {token.strip(): i for i, token in enumerate(f.readline())}

        for token in special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        self.vocab = vocab

    def to_txt(self, file_path):
        sequential_tokens = list(zip(*sorted(self._vocab.items(), key=lambda x: x[1], reverse=False)))[0]
        safe_save(file_path, '\n'.join(sequential_tokens), suffix='txt')

    def to_json(self, file_path):
        safe_save_json(file_path, self._vocab)

    @classmethod
    def read_txt(cls,
                 file_path,
                 encoding='utf-8',
                 token_start: typing.Union[bool, str] = False,
                 token_end: typing.Union[bool, str] = False,
                 token_pad: typing.Union[bool, str] = False,
                 token_unknown: typing.Union[bool, str] = False,
                 token_mask: typing.Union[bool, str] = False,
                 ignore_case=False, ):
        path = pathlib.Path(file_path)
        with open(path, 'r', encoding=encoding) as f:
            vocab = {token.strip(): i for i, token in enumerate(f.readline())}

        instance = cls(
            token_start=token_start,
            token_end=token_end,
            token_pad=token_pad,
            token_unknown=token_unknown,
            token_mask=token_mask,
            ignore_case=ignore_case,
        )
        instance.vocab = vocab
        return instance

    @classmethod
    def read_json(cls,
                  file_path,
                  encoding='utf-8',
                  token_start: typing.Union[bool, str] = False,
                  token_end: typing.Union[bool, str] = False,
                  token_pad: typing.Union[bool, str] = False,
                  token_unknown: typing.Union[bool, str] = False,
                  token_mask: typing.Union[bool, str] = False,
                  ignore_case=False):
        path = pathlib.Path(file_path)
        check_assert_suffix(path, suffix='json')

        with open(path, 'r', encoding=encoding) as f:
            vocab = json.load(f)

        instance = cls(
            token_start=token_start,
            token_end=token_end,
            token_pad=token_pad,
            token_unknown=token_unknown,
            token_mask=token_mask,
            ignore_case=ignore_case,
        )
        instance.vocab = vocab
        return instance

    def fit(self,
            tokens=None,
            corpus_path=None,
            vocab_path=None,
            encoding='utf-8',
            special_tokens=None):
        if tokens is not None:
            self.init_with_tokens(tokens, special_tokens=special_tokens)
        elif vocab_path is not None:
            self.init_with_vocab_file(vocab_path, encoding=encoding, special_tokens=special_tokens)
        else:
            raise NotImplementedError("Unsupported method initializing a `Vocabulary` instance.")


class BertVocabulary(Vocabulary):
    def __init__(self,
                 vocab_path=None,
                 token_start=SOS,
                 token_end=EOS,
                 encoding: str = 'utf-8',
                 simplified: bool = False,
                 special_tokens=None,
                 ignore_case=False,
                 **kwargs):
        super(BertVocabulary, self).__init__(
            ignore_case=ignore_case,
            token_start=token_start,
            token_end=token_end,
            token_unknown=UNK,
            token_pad=PAD,
            token_mask=MASK,
            **kwargs
        )

        self.cls_token = CLS
        self.sep_token = SEP

        if vocab_path is not None:
            self.init_vocab(vocab_path, encoding=encoding, simplified=simplified, special_tokens=special_tokens)

    def init_vocab(self, vocab_path, encoding='utf-8', simplified=False, special_tokens=None):
        vocab = dict()
        with open(vocab_path, 'r', encoding=encoding) as f:
            for line in f:
                token = line.strip()
                if simplified:
                    if token not in vocab:
                        if len(token) > 1:
                            if all([True if not is_cjk_character(char) and not is_punctuation_character(char) else False
                                    for char in self.stem(token)]):
                                vocab[token] = len(vocab)
                        else:
                            vocab[token] = len(vocab)
                else:
                    vocab[token] = len(vocab)

        if special_tokens is not None:
            for token in special_tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        self._vocab = vocab
        self.update_reverse_vocab()
        self.fitted = True

    @staticmethod
    def stem(token):
        r"""If token is start with `##`, remove it from token.
        """
        return token[2:] if token[:2] == '##' else token

    @staticmethod
    def is_special_token(token):
        return bool(token) and (token[0] == '[') and (token[-1] == ']')
