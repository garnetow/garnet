# coding: utf-8

"""
@File   : focal.py
@Author : garnet
@Time   : 2020/10/20 20:18
"""

import keras
import tensorflow as tf

from ..backend import K


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        loss_1 = K.mean(alpha * K.pow((1. - pt_1), gamma) * K.log(pt_1))
        loss_0 = K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        return -loss_1 - loss_0

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=1.):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
      model.compile(loss=[categorical_focal_loss(alpha=1., gamma=2.)], metrics=["accuracy"], optimizer=adam)
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        r"""
        Args:
            :param y_true: A tensor of the same shape as `y_pred`
            :param y_pred: A tensor resulting from a softmax

        Returns:
            A `Tensor` that contains the softmax cross entropy loss. Its type is the
            same as `logits` and its shape is the same as `labels` except that it does
            not have the last dimension of `labels`.
        """

        if isinstance(alpha, (int, float)):
            _alpha = alpha
        else:
            _shape = K.int_shape(y_pred)
            assert len(alpha) == _shape[-1]
            _alpha = K.constant(alpha, dtype=K.dtype(y_pred))
            for i in range(len(_shape) - 1):
                _alpha = K.expand_dims(_alpha, axis=0)

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = _alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(loss, axis=-1)

    return categorical_focal_loss_fixed


def sparse_categorical_focal_loss(gamma=2., alpha=1.):
    def sparse_categorical_focal_loss_fixed(y_true, y_pred):
        r"""
        Args:
            y_true: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
                `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
                must be an index in `[0, num_classes)`. Other values will raise an
                exception when this op is run on CPU, and return `NaN` for corresponding
                loss and gradient rows on GPU.
            y_pred: Per-label activations (typically a linear output) of shape
                `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32`, or
                `float64`. These activation energies are interpreted as unnormalized log
                probabilities.

        Returns:
            A `Tensor` of the same shape as `labels` and of the same type as `logits`
            with the softmax cross entropy loss.
        """
        if isinstance(alpha, (int, float)):
            _alpha = alpha
        else:
            _shape = K.int_shape(y_pred)
            assert len(alpha) == _shape[-1]
            _alpha = K.constant(alpha, dtype=K.dtype(y_pred))
            for i in range(len(_shape) - 1):
                _alpha = K.expand_dims(_alpha, axis=0)

        # Transfer class index into one-hot sparse labels
        y_true = K.one_hot(K.cast(y_true, dtype='int32'), num_classes=K.int_shape(y_pred)[-1])
        y_true = K.cast(y_true, dtype=K.dtype(y_pred))

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = _alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(loss, axis=-1)

    return sparse_categorical_focal_loss_fixed


custom_objects = {
    'binary_focal_loss': binary_focal_loss,
    'categorical_focal_loss': categorical_focal_loss,
    'sparse_categorical_focal_loss': sparse_categorical_focal_loss,
}

keras.utils.get_custom_objects().update(custom_objects)
