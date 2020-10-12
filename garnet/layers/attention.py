# coding: utf-8

"""
@File   : attention.py
@Author : garnet
@Time   : 2020/10/10 16:53
"""

import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, Dense

from ..backend.mask import sequence_masking


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
        self.output_dim = head_size * head_num
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
             query_mask=None,
             value_mask=None,
             attention_mask=None,  # (batch_size, query_length, key_length)
             head_mask=None,  # (num_heads,)
             encoder_hidden_context=None,
             encode_attention_mask=None,
             **kwargs):
        """
        q: (batch_size, query_seq_len, hidden_query)
        k: (batch_size, key_seq_len, hidden_key)
        v: (batch_size, key_seq_len, hidden_value)
        """
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs

        if query_mask is not None:
            query_mask = K.cast(query_mask, dtype=K.floatx())
        if value_mask is not None:
            value_mask = K.cast(value_mask, dtype=K.floatx())

        # linear transformation
        qw = self.q_proj(q)  # (batch_size, query_seq_len, head_num * key_size)
        kw = self.k_proj(k)  # (batch_size, key_seq_len, head_num * key_size)
        vw = self.v_proj(v)  # (batch_size, key_seq_len, head_num * head_size)

        # shape transposing for calculating score
        qw = K.reshape(
            qw,
            (-1, K.shape(q)[1], self.head_num, self.key_size)
        )  # (batch_size, query_seq_len, head_num, key_size)
        kw = K.reshape(
            kw,
            (-1, K.shape(k)[1], self.head_num, self.key_size)
        )  # (batch_size, key_seq_len, head_num, key_size)
        vw = K.reshape(
            vw,
            (-1, K.shape(v)[1], self.head_num, self.head_size)
        )  # (batch_size, key_seq_len, head_num, head_size)

        # calculate attention scores(scale-dot method)
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)  # (batch_size, head_num, query_seq_len, key_seq_len)
        if self.attention_scale:
            a = a / (self.key_size ** 0.5)

        # apply value sequence mask
        a = sequence_masking(a, value_mask, mode='add', axis=-1)

        # apply attention mask
        if attention_mask is not None:
            att_mask_dim = K.ndim(attention_mask)
            if att_mask_dim == 3:
                attention_mask = K.expand_dims(attention_mask, axis=1)
            a -= (1 - attention_mask) * 1e12

        # apply softmax on each query sequence steps
        a = K.softmax(a, axis=-1)

        # apply head mask
        if head_mask is not None:
            # head_mask with shape (num_heads,)
            head_mask = K.expand_dims(K.expand_dims(K.expand_dims(head_mask, axis=0), axis=-1), axis=-1)
            a = a * head_mask

        # get output
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)  # (batch_size, query_seq_len, head_num, head_size)
        o = K.reshape(o, shape=(-1, K.shape(o)[1], self.output_dim))  # (batch_size, query_seq_len, head_num*head_size)
        o = self.o_proj(o)

        # apply query sequence mask
        o = sequence_masking(o, query_mask, mode='mul', axis=1)
        return o

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            batch_size, sequence_length = input_shape[0][0], input_shape[0][1]
        else:
            batch_size, sequence_length = input_shape[0], input_shape[1]
        return batch_size, sequence_length, self.output_dim

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return mask[0]
        return mask

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'head_size': self.head_size,
            'output_dim': self.output_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'MultiHeadAttention': MultiHeadAttention,
}

keras.utils.get_custom_objects().update(custom_objects)
