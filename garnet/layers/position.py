# coding: utf-8

"""
@File   : position.py
@Author : garnet
@Time   : 2020/11/18 11:33
"""

import keras
import keras.backend as K
from keras.layers import Layer


class RelativePositionEmbedding(Layer):
    r"""Relative position embedding layer.

    See more from paper: https://arxiv.org/abs/1803.02155
    """

    def __init__(self, input_dim, output_dim, embeddings_initializer='zeros', **kwargs):
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)

        self.embeddings = None

    def build(self, input_shape):
        super(RelativePositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
        )

    def call(self, inputs, **kwargs):
        relative = self.compute_position_ids(inputs)
        return K.gather(self.embeddings, relative)

    def compute_position_ids(self, inputs):
        q, k = inputs

        q_indices = K.arange(0, K.shape(q)[1], dtype='int32')
        q_indices = K.expand_dims(q_indices, axis=1)
        k_indices = K.arange(0, K.shape(k)[1], dtype='int32')
        k_indices = K.expand_dims(k_indices, axis=0)
        relative = q_indices - k_indices

        # truncate and transfer into positive indices
        max_index = (self.input_dim - 1) // 2
        relative = K.clip(relative, -max_index, max_index)
        return relative + max_index

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'RelativePositionEmbedding': RelativePositionEmbedding,
}

keras.utils.get_custom_objects().update(custom_objects)
