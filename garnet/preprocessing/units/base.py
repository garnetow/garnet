# coding: utf-8

"""
@File   : base.py
@Author : garnet
@Time   : 2020/10/14 23:11
"""

import abc


class Unit(metaclass=abc.ABCMeta):
    r"""Data/text process unit without states, in other words it does not need fitting.
    """

    @abc.abstractmethod
    def transform(self, inputs, *args, **kwargs):
        ...


class StatefulUnit(Unit, metaclass=abc.ABCMeta):
    r"""Data/text process unit with states. Need to be fitted before transformation. All states will be gathered during
    fitting process.
    """

    def __init__(self, *args, **kwargs):
        self.fitted = False

    def fit(self, *args, **kwargs):
        self.fitted = True

    def reverse_transform(self, *args, **kwargs):
        ...
