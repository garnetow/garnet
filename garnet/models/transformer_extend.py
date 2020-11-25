# coding: utf-8

"""
@File   : transformer_extend.py
@Author : garnet
@Time   : 2020/11/24 11:37
"""

from .transformer_mixin import UniLMMixin
from .transformer_mixin import LanguageModelMixin


def extend_with_unified_language_model(ModelClass):
    class UnifiedLanguageModel(UniLMMixin, ModelClass):
        def __init__(self, *args, **kwargs):
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = True

    return UnifiedLanguageModel


def extend_with_language_model(ModelClass):
    class LanguageModel(LanguageModelMixin, ModelClass):
        def __init__(self, *args, **kwargs):
            super(LanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = True
