# coding: utf-8

"""
@File   : transformer_mixin.py
@Author : garnet
@Time   : 2020/11/24 11:35
"""

import keras.backend as K
from keras.layers import Lambda

from ..backend import unilm_mask
from ..backend import language_model_mask


class UniLMMixin(object):
    r"""Register compute UniLM attention mask method into transformer-like model class.

    See more at https://arxiv.org/abs/1905.03197 and https://kexue.fm/archives/6933#Seq2Seq.
    """

    def compute_attention_mask(self, inputs=None, seg_ids=None, **kwargs):
        r"""
        Argument:
            :param inputs: segment ids with shape (batch_size, seq_length)
        """
        if self.attention_mask is None:
            self.attention_mask = self.apply(
                inputs=seg_ids,
                layer=Lambda,
                function=unilm_mask,
                name='Attention-UniLM-Mask'
            )
        return self.attention_mask


class LanguageModelMixin(object):
    r"""Register compute lower triangular attention mask method into transformer-like model class, which used for
    language model.
    """

    def compute_attention_mask(self, inputs=None, **kwargs):
        if self.attention_mask is None:
            def mask(x):
                mask = language_model_mask(x)
                return K.expand_dims(mask, axis=1)

            self.attention_mask = self.apply(
                inputs=inputs,
                layer=Lambda,
                function=mask,
                name='Attention-LM-Mask'
            )
        return self.attention_mask
