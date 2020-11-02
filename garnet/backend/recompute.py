# coding: utf-8

"""
@File   : recompute.py
@Author : garnet
@Time   : 2020/11/2 11:40
"""

import os
import tensorflow as tf
from distutils.util import strtobool
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.ops.custom_gradient import _graph_mode_decorator


def recompute_grad(call):
    r"""Decorator of recomputing forward pass.

    This function is used to decorate :meth:`call` of Layer in keras when define custom layers. The outputs of
    decorated layers will be forget in forward pass, and undecorated layer will be the checkpoint to recompute outputs
    in backward pass.

    See more about recomputing at: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174).
    """
    if not strtobool(os.environ.get('RECOMPUTE', '0')):
        return call

    def inner(self, inputs, **kwargs):
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            r"""Forward pass.
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            r"""Forward and backward pass.
            """
            outputs = kernel_call()

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        return _graph_mode_decorator(call_and_grad, *flat_inputs)

    return inner
