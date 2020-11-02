# coding: utf-8

"""
@File   : sequence.py
@Author : garnet
@Time   : 2020/10/22 17:06
"""

import numpy as np


def sequence_padding(seq, max_len=None, padding='post', truncate='post', padding_index=0, dtype='int32'):
    r"""Truncate or pad sequence into fixed length.
    """

    max_len = max_len or max(len(s) for s in seq)

    outputs = []
    for x in seq:
        x = x[:max_len] if truncate == 'post' else x[-max_len:]
        if padding == 'post':
            x += [padding_index] * (max_len - len(x))
        else:
            x = [padding_index] * (max_len - len(x)) + x
        outputs.append(x)
    outputs = np.array(outputs, dtype=dtype)
    return outputs
