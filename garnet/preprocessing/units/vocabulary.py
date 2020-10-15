# coding: utf-8

"""
@File   : vocabulary.py
@Author : garnet
@Time   : 2020/10/15 8:21
"""

import typing
from .base import StatefulUnit

PAD = '[PAD]'
UNK = '[UNK]'
SOS = '[SOS]'  # start of sentence
EOS = '[EOS]'  # end of sentence
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'


def load_text(text_files, encoding='utf-8', lang='eng'):
    token_list = []

    with open(text_files, 'r', encoding=encoding) as f:
        for line in f:
            line.strip()


class Vocabulary(StatefulUnit):
    def __init__(self,
                 vocab_path=None,
                 corpus_path=None,
                 token_start: typing.Optional[bool, str] = False,
                 token_end: typing.Optional[bool, str] = False,
                 token_pad: typing.Optional[bool, str] = False,
                 token_unknown: typing.Optional[bool, str] = False,
                 token_mask: typing.Optional[bool, str] = False,
                 special_tokens=None,
                 ignore_case=False,
                 **kwargs):
        super(Vocabulary, self).__init__(**kwargs)
        self.ignore_case = ignore_case
        self._token_start = token_start
        self._token_end = token_end
        self._token_pad = token_pad
        self._token_mask = token_mask
        self._token_unk = token_unknown

        self._vocab = dict()
        self._vocab_reverse = dict()

        if vocab_path is not None:
            self.load_vocab(vocab_path)
        elif corpus_path is not None:
            pass

    def load_vocab(self, vocab_path, encoding='utf-8', simplified=False, special_tokens=None):
        vocab = dict()
        with open(vocab_path, 'r', encoding=encoding) as f:
            for line in f:
                token = line.strip()
                if simplified:
                    pass
                else:
                    vocab[token] = len(vocab)

    def fit(self, inputs, special_tokens=None, overwrite=False):
        pass
