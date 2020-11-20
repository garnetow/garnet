# coding: utf-8

"""
@File   : adafactor.py
@Author : garnet
@Time   : 2020/10/21 8:36
"""

import keras
import keras.backend as K
import numpy as np


class AdaFactorOptimizer(keras.optimizers.Optimizer):
    """
    AdaFactor optimizer class.
    From paper: [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
    Refer to: https://github.com/bojone/adafactor

    learning_rate: If `learning_rate` is `None`, then dynamic learning rate is min(10^{-2}, 1/\\sqrt{t})
    beta1: Decay hyper parameter of momentum, default is 0.0, then momentum is not used while optimizing,
        to save memory, which is recommended. If `beta1` is greater than 0.0, momentum is used like Adam
    beta2: Decay hyper parameter of second-order momentum, default is `None`, then dynamic second-order momentum is
        used, which is recommended. If `beta2` is not `None`, beta2 is fixed while training
    epsilon1: Regularization constant for squared gradient
    epsilon2: Regularization constant for parameter scale
    clipping_threshold: Hyper parameter of gradient truncation
    multiply_by_parameter_scale: Whether scale learning rate. Only available when `learning_rate` is `None`, and
        dynamic learning rate is used. Default is `False`
    min_dim_size_to_factor: Only factor accumulator if two tensor dimensions are at least this size
    """

    def __init__(self, learning_rate=None, beta1=0.0, beta2=None, epsilon1=1e-30, epsilon2=1e-3, clipping_threshold=1.0,
                 multiply_by_parameter_scale=False, min_dim_size_to_factor=128, **kwargs):
        super(AdaFactorOptimizer, self).__init__(**kwargs)
        self._learning_rate = learning_rate
        self.beta1 = beta1
        self._beta2 = beta2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.clipping_threshold = clipping_threshold
        self.min_dim_size_to_factor = min_dim_size_to_factor
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

    @property
    def learning_rate(self):
        if self._learning_rate is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            learning_rate = K.minimum(1.0 / K.sqrt(iterations), 0.01)
            if self.multiply_by_parameter_scale:
                return learning_rate
            else:
                return learning_rate * 0.05
        else:
            if not hasattr(self, '__learning_rate'):
                with K.name_scope(self.__class__.__name__):
                    self.__learning_rate = K.variable(self._learning_rate, name='learning_rate')
            return self.__learning_rate

    @property
    def beta2(self):
        if self._beta2 is None:
            iterations = K.cast(self.iterations + 1, K.floatx())
            return 1.0 - K.pow(iterations, -0.8)
        else:
            return self._beta2

    def factored_shape(self, shape):
        """
        Decompose weight matrix(tensor) into two low rank matrix(tensor), this function calculates the shapes of these
        two new matrix.
        For 2d weight matrix with shape (m, n), two low rank matrix will be (m, 1) and (1, n).
        For weight tensor with more than 2 dimension, in order to get least parameters, use `np.argpartition` get the
        longest dimension, decomposition will be performed on last two dimensions

        shape: Shape of weight tensor
        """
        if len(shape) < 2:
            return None

        shape = np.array(shape)
        indices = shape.argpartition(-2)
        if indices[-2] < self.min_dim_size_to_factor:
            return None

        shape1, shape2 = np.array(shape), np.array(shape)  # shape of two low rank tensor
        shape1[indices[-1]] = 1
        shape2[indices[-2]] = 1
        return shape1, indices[-1], shape2, indices[-2]

    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.weights = [self.iterations]

        lr = self.learning_rate
        for i, (p, g) in enumerate(zip(params, grads)):
            g2 = K.square(g) + self.epsilon1  # squared gradients

            # calculate smooth squared gradients
            shape, dtype = K.int_shape(p), K.dtype(p)
            factored_shape = self.factored_shape(shape)
            if factored_shape is None:  # tensor decomposition will not be applied
                # define smooth squared gradients
                v = K.zeros(shape, dtype=dtype, name='v_'.format(i))
                self.weights.append(v)
                # define update process
                v_t = self.beta2 * v + (1 - self.beta2) * g2
                self.updates.append(K.update(v, v_t))
            else:
                # define smooth squared gradients
                shape1, axis1, shape2, axis2 = factored_shape
                vr = K.zeros(shape1, dtype=dtype, name='vr_'.format(i))
                vc = K.zeros(shape2, dtype=dtype, name='vc_'.format(i))
                self.weights.extend([vr, vc])
                # define update process
                vr_t = self.beta2 * vr + (1 - self.beta2) * K.sum(g2, axis=axis1, keepdims=True)
                vc_t = self.beta2 * vc + (1 - self.beta2) * K.sum(g2, axis=axis2, keepdims=True)
                self.updates.extend([K.update_add(vr, vr_t), K.update(vc, vc_t)])
                v_t = vr_t * vc_t / K.mean(vr_t, axis=axis2, keepdims=True)
            u = g / K.sqrt(v_t)
            # gradient truncate
            if self.clipping_threshold is not None:
                u_rms = K.sqrt(K.mean(K.square(u)))
                u /= K.maximum(1., u_rms / self.clipping_threshold)
            # gradient momentum
            if self.beta1 is not None and self.beta1 > 0.0:
                m = K.zeros(shape, dtype=dtype, name='m_'.format(i))
                self.weights.append(m)
                m_t = self.beta1 * m + (1.0 - self.beta1) * u
                self.updates.append(K.update(m, m_t))
            if self.multiply_by_parameter_scale:
                u = u * K.maximum(self.epsilon2, K.sqrt(K.mean(K.square(p))))
            self.updates.append(K.update(p, p - lr * u))
        return self.updates

    def get_config(self):
        config = {
            'learning_rate': self._learning_rate,
            'beta1': self.beta1,
            'beta2': self._beta2,
            'epsilon1': self.epsilon1,
            'epsilon2': self.epsilon2,
            'multiply_by_parameter_scale': self.multiply_by_parameter_scale,
            'clipping_threshold': self.clipping_threshold,
            'min_dim_size_to_factor': self.min_dim_size_to_factor,
        }
        base_config = super(AdaFactorOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'AdaFactorOptimizer': AdaFactorOptimizer,
}

keras.utils.get_custom_objects().update(custom_objects)
