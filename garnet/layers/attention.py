# coding: utf-8

"""
@File   : attention.py
@Author : garnet
@Time   : 2020/10/10 16:53
"""

import keras
import keras.backend as K
from keras.layers import Layer, Dense

"""
def __init__(
        self,
        heads,
        head_size,
        key_size=None,
        use_bias=True,
        attention_scale=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
"""


class MultiHeadAttention(Layer):
    r"""Multi-head attention layer. See: https://arxiv.org/pdf/1706.03762.pdf

    Argument:
        :param head_num: Number of heads.
        :param head_size: output size of single head.
        :param key_size (int, optional, default: `None`): internal hidden size of query and key vector.
        :param use_bias (bool, optional, default: `True`): Whether to use bias term.
        :param attention_scale (bool, optional, default: `True`): whether apply scale on attention matrix.
        :param activation: Activations for linear mappings.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
    """

    def __init__(self,
                 head_num,
                 head_size,
                 key_size=None,
                 use_bias=True,
                 attention_scale=True,
                 activation=None,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supports_masking = True

        self.head_num = head_num
        self.head_size = head_size
        self.output_dim = head_size * head_num if head_size else None
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale

        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.q_proj = self.k_proj = self.v_proj = self.o_proj = None

    def build(self, input_shape):
        self.q_proj = Dense(
            units=self.key_size * self.head_num,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.k_proj = Dense(
            units=self.key_size * self.head_num,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.v_proj = Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.o_proj = Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self,
             inputs,
             attention_mask=None,  # (batch_size, query_length, key_length)
             head_mask=None,  # (num_heads,) or (num_layers, num_heads)
             encoder_hidden_context=None,
             encode_attention_mask=None,
             **kwargs):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
