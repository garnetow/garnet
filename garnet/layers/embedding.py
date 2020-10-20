# coding: utf-8

"""
@File   : embedding.py
@Author : garnet
@Time   : 2020/10/13 10:43
"""

import keras
import keras.backend as K
from keras.layers import Embedding


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


custom_objects = {
    'DenseEmbedding': DenseEmbedding,
}

keras.utils.get_custom_objects().update(custom_objects)
