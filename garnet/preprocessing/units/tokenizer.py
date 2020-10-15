# coding: utf-8

"""
@File   : tokenizer.py
@Author : garnet
@Time   : 2020/10/14 23:50
"""

import typing

from .base import StatefulUnit


class BaseTokenizer(StatefulUnit):
    def __init__(self,
                 vocab: typing.Optional[str],
                 **kwargs):
        super(BaseTokenizer, self).__init__(**kwargs)
