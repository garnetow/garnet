# coding: utf-8

"""
@File   : bert.py
@Author : garnet
@Time   : 2020/11/2 11:39
"""

import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, Dense, Embedding

from ..backend.mask import sequence_masking
from ..backend.recompute import recompute_grad


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

    @recompute_grad
    def call(self,
             inputs,
             query_mask=None,
             value_mask=None,
             attention_mask=None,  # (batch_size, query_length, key_length) or (batch_size, 1, query_length, key_length)
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


class LayerNormalization(Layer):
    r"""Conditional layer normalization.

    See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

    Parameters with prefix `cond_hidden_*` are available it is a conditional layer, and control the behavior of
    condition units.

    Argument:
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param conditional (bool, optional, default: False): being a conditional layer normalization or not.
        :param cond_hidden_units: (int, optional, default: None):
            number of units in hidden layer which projects input of conditions into `cond_hidden_units` shape.
        :param cond_hidden_activation: activation of condition hidden layer.
        :param cond_hidden_initializer: initializer of condition hidden layer.
    """

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 conditional=False,
                 cond_hidden_units=None,
                 cond_hidden_activation='linear',
                 cond_hidden_initializer='glorot_uniform',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True

        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)

        self.conditional = conditional
        self.cond_hidden_units = cond_hidden_units
        self.cond_hidden_activation = keras.activations.get(cond_hidden_activation)
        self.cond_hidden_initializer = keras.initializers.get(cond_hidden_initializer)

        self.gamma, self.beta = None, None
        self.cond_hidden, self.cond_beta, self.cond_gamma = None, None, None

    def compute_output_shape(self, input_shape):
        return input_shape[0] if self.conditional else input_shape

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = [K.expand_dims(m, 0) for m in mask if m is not None]
            return None if len(masks) == 0 else K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        shape = input_shape[0][-1:] if self.conditional else input_shape[-1:]

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )

        if self.conditional:
            if self.cond_hidden_units is not None:
                self.cond_hidden = Dense(
                    units=self.cond_hidden_units,
                    kernel_initializer=self.cond_hidden_initializer,
                    activation=self.cond_hidden_activation,
                    use_bias=False,
                )
            if self.center:
                self.cond_beta = Dense(
                    units=shape[0],
                    kernel_initializer='zeros',
                    use_bias=False,
                )
            if self.scale:
                self.cond_gamma = Dense(
                    units=shape[0],
                    kernel_initializer='zeros',
                    use_bias=False,
                )

        super(LayerNormalization, self).build(input_shape)

    @recompute_grad
    def call(self, inputs, **kwargs):
        beta, gamma = None, None
        if self.conditional:
            inputs, cond = inputs
            if self.cond_hidden_units is not None:
                cond = self.cond_hidden(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta + self.cond_beta(cond)
            if self.scale:
                gamma = self.gamma + self.cond_gamma(cond)
        else:
            beta = self.beta
            gamma = self.gamma

        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std

        if self.scale:
            outputs *= gamma
        if self.center:
            outputs += beta

        return outputs

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'conditional': self.conditional,
            'cond_hidden_units': self.cond_hidden_units,
            'cond_hidden_activation': keras.activations.serialize(self.cond_hidden_activation),
            'cond_hidden_initializer': keras.initializers.serialize(self.cond_hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    r"""Feed forward layer, equivalent to two consecutive dense layer.

    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.supports_masking = True

        self.units = units
        self.use_bias = use_bias

        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.dense1, self.dense2 = None, None

    def build(self, input_shape):
        output_dim = input_shape[-1]

        self.dense1 = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.dense2 = Dense(
            units=output_dim,
            activation=None,  # no activation in the second dense layer
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )

        super(FeedForward, self).build(input_shape)

    @recompute_grad
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'use_bias': self.use_bias,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseEmbedding(Embedding):
    r"""Extended Embedding.

    Extend a new forward process mode to normal embedding layer. There are two mode now:
    `embedding`: normal embedding process, accept integer tensor as input, transform integer into corresponding vector.
    `dense`: equivalent to dense, the length of last dimension of input is the same as embedding size, multiply input
        tensor and embedding matrix, and output is the logit of each token in the vocabulary. Usually used in Masked
        Language Model prediction task.
    """

    def __init__(self, *args, **kwargs):
        super(DenseEmbedding, self).__init__(*args, **kwargs)
        self._current_mode = 'embedding'

    def call(self, inputs, mode='embedding'):
        """In keras framework, `call` method is first called, followed by `compute_mask`,
        and then `compute_output_shape`
        """
        self._current_mode = mode
        if mode == 'embedding':
            return super(DenseEmbedding, self).call(inputs)
        else:
            return K.dot(inputs, K.transpose(self.embeddings))

    def compute_mask(self, inputs, mask=None):
        if self._current_mode == 'embedding':
            mask = super(DenseEmbedding, self).compute_mask(inputs, mask)
            if mask is not None:
                # make sure [CLS] token will not be masked
                cls_mask = K.ones_like(mask[:, :1], dtype='bool')
                other_mask = mask[:, 1:]
                return K.concatenate([cls_mask, other_mask], axis=1)
        else:
            return mask

    def compute_output_shape(self, input_shape):
        if self._current_mode == 'embedding':
            return super(DenseEmbedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:-1] + (self.input_dim,)


class PositionEmbedding(Layer):
    r"""Trainable position embedding.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 merge_mode='add',
                 embeddings_initializer='zeros',
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(name='embeddings', shape=(self.input_dim, self.output_dim),
                                          initializer=self.embeddings_initializer)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        custom_position_ids = None
        if isinstance(inputs, list) and len(inputs) == 2:
            inputs, custom_position_ids = inputs

        if custom_position_ids is None:
            input_shape = K.shape(inputs)
            batch_size, seq_length = input_shape[0], input_shape[1]
            pos_embeddeing = self.embeddings[:seq_length]
            pos_embeddeing = K.expand_dims(pos_embeddeing, axis=0)
            if self.merge_mode != 'add':
                pos_embeddeing = K.tile(pos_embeddeing, n=[batch_size, 1, 1])
        else:
            pos_embeddeing = K.gather(self.embeddings, custom_position_ids)

        if self.merge_mode == 'add':
            return inputs + pos_embeddeing
        else:
            return K.concatenate([inputs, pos_embeddeing], axis=0)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) == 2:
            input_shape, _ = input_shape

        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
    'FeedForward': FeedForward,
    'DenseEmbedding': DenseEmbedding,
    'PositionEmbedding': PositionEmbedding,
}

keras.utils.get_custom_objects().update(custom_objects)
