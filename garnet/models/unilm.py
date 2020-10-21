# coding: utf-8

"""
@File   : unilm.py
@Author : garnet
@Time   : 2020/10/21 10:10
"""

from keras.layers import Lambda

from ..backend import unilm_mask


class UniLMMixin(object):
    r"""Register compute UniLM attention mask method into normal model class.

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


def extend_with_unified_language_model(ModelClass):
    class UnifiedLanguageModel(UniLMMixin, ModelClass):
        def __init__(self, *args, **kwargs):
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = True

    return UnifiedLanguageModel
