# coding: utf-8

"""
@File   : t5.py
@Author : garnet
@Time   : 2020/11/18 11:32
"""

import keras
import numpy as np

from ..backend import K
from .position import RelativePositionEmbedding


class RelativePositionEmbeddingT5(RelativePositionEmbedding):
    def __init__(self,
                 input_dim,
                 output_dim,
                 max_distance=128,
                 bidirectional=True,
                 embeddings_initializer='zeros',
                 **kwargs):
        super(RelativePositionEmbeddingT5, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            **kwargs
        )
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def compute_position_ids(self, inputs):
        q, k = inputs

        q_indices = K.arange(0, K.shape(q)[1], dtype='int32')
        q_indices = K.expand_dims(q_indices, axis=1)
        k_indices = K.arange(0, K.shape(k)[1], dtype='int32')
        k_indices = K.expand_dims(k_indices, axis=0)
        relative = q_indices - k_indices

        num_buckets = self.input_dim
        final_indices = 0
        n = -relative
        if self.bidirectional:
            num_buckets //= 2
            final_indices += K.cast(K.less(n, 0), 'int32') * num_buckets
            n = K.abs(n)
        else:
            n = K.maximum(n, 0)

        max_exact = num_buckets // 2
        is_small = K.less(n, max_exact)
        val_if_large = max_exact + K.cast(
            K.log(K.cast(n, K.floatx()) / max_exact) /
            np.log(self.max_distance / max_exact) * (num_buckets - max_exact),
            'int32',
        )
        val_if_large = K.minimum(val_if_large, num_buckets - 1)
        final_indices += K.switch(is_small, n, val_if_large)
        return final_indices

    def get_config(self):
        config = {
            'max_distance': self.max_distance,
            'bidirectional': self.bidirectional,
        }
        base_config = super(RelativePositionEmbeddingT5, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'RelativePositionEmbeddingT5': RelativePositionEmbeddingT5,
}

keras.utils.get_custom_objects().update(custom_objects)
