# coding: utf-8

"""
@File   : feedforward.py
@Author : garnet
@Time   : 2020/10/12 11:41
"""

import keras
import keras.backend as K
from keras.layers import Layer, Dense, Dropout


class FeedForward(Layer):
    r"""Feed forward layer, equivalent to two consecutive dense layer.

    """

    def __init__(self,
                 units,
                 activation='relu',
                 dropout_rate=0.0,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.supports_masking = True

        self.units = units
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.dense1, self.dense2 = None, None

    def build(self, input_shape):
        output_dim = input_shape[-1]

        self.dense1 = Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )
        self.dense2 = Dense(
            units=output_dim,
            activation=None,  # no activation in the second dense layer
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
        )

        super(FeedForward, self).build(input_shape)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        if self.dropout_rate > 0.:
            def dropped_inputs():
                return K.dropout(x, self.dropout_rate)

            x = K.in_train_phase(dropped_inputs, x, training=training)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'FeedForward': FeedForward,
}

keras.utils.get_custom_objects().update(custom_objects)
