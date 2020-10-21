# coding: utf-8

"""
@File   : simbert.py
@Author : garnet
@Time   : 2020/10/21 15:25
"""

import keras
import keras.backend as K

from .loss import LossLayer


class SimBertLoss(LossLayer):
    r"""The loss of SimBert model is combined by two component:

    1. cross-entropy loss of seq2seq model
    2. cross-entropy of similar sentences in a batch
    """

    def __init__(self, cls_logit_scale=30, **kwargs):
        super(SimBertLoss, self).__init__(**kwargs)
        self.cls_logit_scale = cls_logit_scale

    def compute_loss(self, inputs, mask=None):
        r"""
        Args:
            :param inputs: a list of length 4.
                inputs[0]: (batch_size, seq_length) raw token ids.
                inputs[1]: (batch_size, seq_length) segment ids. `0` means first sentence and `1` means second.
                inputs[2]: (batch_size, hidden_size) output feature of [CLS].
                inputs[3]: (batch_size, seq_length, vocab_size) token predict probas.
        """
        similarity_loss = self.compute_cls_similarity_loss(inputs, mask=mask)
        seq2seq_loss = self.compute_seq2seq_loss(inputs, mask=mask)
        self.add_metric(similarity_loss, name='similarity_loss')
        self.add_metric(seq2seq_loss, name='seq2seq_loss')
        return seq2seq_loss + similarity_loss

    def compute_cls_similarity_loss(self, inputs, mask=None):
        _, _, cls_output, _ = inputs
        y_true = self._batch_similar_label(cls_output)
        norm_cls = K.l2_normalize(cls_output, axis=1)
        y_pred = K.dot(norm_cls, K.transpose(norm_cls))
        y_pred -= K.eye(K.shape(y_pred)[0]) * 1e12  # mask diagonal
        y_pred *= self.cls_logit_scale
        loss = K.categorical_crossentropy(target=y_true, output=y_pred, from_logits=True)
        return loss

    def compute_seq2seq_loss(self, inputs, mask=None):
        token_ids, seg_ids, _, y_pred = inputs
        y_true = token_ids[:, 1:]  # target ids
        y_mask = seg_ids[:, 1:]
        y_pred = y_pred[:, :-1]  # predict logits
        loss = K.categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def _batch_similar_label(self, batch_tensor):
        batch_index = K.arange(0, K.shape(batch_tensor)[0])
        index1 = K.expand_dims(batch_index, axis=0)
        index2 = K.expand_dims((batch_index + 1 - batch_index % 2 * 2), axis=1)  # neighbor samples are similar samples
        return K.cast(K.equal(index1, index2), dtype=K.floatx())

    def get_config(self):
        config = {
            'cls_logit_scale': self.cls_logit_scale,
        }
        base_config = super(SimBertLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'SimBertLoss': SimBertLoss,
}

keras.utils.get_custom_objects().update(custom_objects)
