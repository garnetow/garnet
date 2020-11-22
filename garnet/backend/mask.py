# coding: utf-8

"""
@File   : mask.py
@Author : garnet
@Time   : 2020/10/12 8:48
"""

import keras.backend as K


def sequence_masking(seq, mask=None, mode='add', axis=1):
    """Mask sequence tensor along step dimension.

    Argument:
        :param seq: sequence tensor.
        :param mask: mask tensor with shape like (batch_size, seq_length). All value in this tensor
            must be either 0 or 1, and 0 means steps will be masked.
        :param mode (str, optional, default: `'add'`): masking method. Applying two method:
            mul: multiply sequence tensor with mask tensor, and `0` in the mask will eliminate corresponding steps;
            add: subtract a huge positive value on steps which corresponding position in mask are `0`.
        :param axis (int, optional, defalut: `1`): step axis in `seq` tensor.
    """
    if mask is None:
        return seq

    if axis is None:
        axis = 1

    x_dims, mask_dims = K.ndim(seq), K.ndim(mask)
    if isinstance(axis, int) and axis < 0:
        if abs(axis) > x_dims:
            raise ValueError("sequence tensor has {} dimensions, but given step dimension is {}".format(x_dims, axis))
        axis = x_dims + axis

    for _ in range(axis - 1):
        mask = K.expand_dims(mask, axis=1)
    for _ in range(x_dims - mask_dims - axis + 1):
        mask = K.expand_dims(mask, axis=-1)

    if mode.lower() == 'mul':
        return seq * mask
    elif mode.lower() == 'add':
        return seq - (1 - mask) * 1e12
    else:
        raise ValueError("`mode` must one of (`mul`, 'add'), got {} instead".format(mode))


def language_model_mask(x):
    r"""Lower triangular attention mask, used for language model.

    Position below diagonal will be marked as `1`, which means not masked.

    Argument:
        :param x: segment ids with shape (batch_size, seq_length)

    Return:
        mask tensor with shape (1, 1, seq_length, seq_length), and second dimension usually means head nums
        in Bert-like model.
    """
    seq_length = K.shape(x)[1]
    indices = K.arange(0, seq_length)
    mask = indices[None, :] <= indices[:, None]
    mask = K.cast(mask, K.floatx())
    return mask[None, None]


def unilm_mask(x):
    r"""UniLM attention mask, used for seq2seq model.

    See more at https://arxiv.org/abs/1905.03197.

    Argument:
        :param x: segment ids with shape (batch_size, seq_length)

    Return:
        mask tensor with shape (batch_size, 1, seq_length, seq_length), and second dimension usually means head nums
        in Bert-like model.
    """
    index = K.cumsum(x, axis=1)
    mask = index[:, None, :] <= index[:, :, None]
    mask = K.cast(mask, K.floatx())
    return mask[:, None]
