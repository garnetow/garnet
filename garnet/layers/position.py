# coding: utf-8

"""
@File   : position.py
@Author : garnet
@Time   : 2020/9/30 16:47
"""

import keras
import keras.backend as K
from keras.layers import Layer


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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(name='embeddings', shape=(self.input_dim, self.output_dim),
                                          initializer=self.embeddings_initializer)
        super().build(input_shape)

    def call(self, inputs, custom_position_ids=None, **kwargs):
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
        pass
