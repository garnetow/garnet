# coding: utf-8

"""
@File   : transformer.py
@Author : garnet
@Time   : 2020/9/29 8:48
"""

from .model import WrappedModel


class Transformer(WrappedModel):
    def __init__(
            self,
            vocab_size,  # 词表大小
            hidden_size,  # 编码维度
            num_hidden_layers,  # 隐层数
            num_attention_heads,  # Multi-head attention中head的数量
            intermediate_size,  # FeedForward的隐层维度
            hidden_act='gelu',  # FeedForward隐层的激活函数
            attention_probs_dropout_prob=None,  # Attention层dropout比例
            hidden_dropout_prob=None,  # 其他层dropout比例
    ):
        r"""Transformer model.

        Arguments:
            :param vocab_size: vocabulary size.
            :param hidden_size: encoding size of token.
            :param num_hidden_layers: number of hidden layers in the Transformer encoder.
            :param num_attention_heads: number of attention heads for each attention layer in the Transformer encoder.
            :param intermediate_size: dimensionality of the feed-forward layer in the Transformer encoder.
            :param hidden_act (:obj:`str` or :obj:`Callable`, optional, default: `'gelu'`):
                non-linear activation function in the encoder and pooler.
            :param attention_probs_dropout_prob (:obj:`float`, optional, default: `None`):
                the dropout ratio for the attention probabilities.
            :param hidden_dropout_prob (:obj:`float`, optional, default: `None`):
                the dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

        """
        pass
