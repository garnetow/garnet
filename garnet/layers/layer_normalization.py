# coding: utf-8

"""
@File   : layer_normalization.py
@Author : garnet
@Time   : 2020/10/6 9:28
"""

import keras
import keras.backend as K
from keras.layers import Layer, Dense


class LayerNormalization(Layer):
    r"""Conditional layer normalization.

    See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

    Parameters with prefix `cond_hidden_*` are available it is a conditional layer, and control the behavior of
    condition units.

    Argument:
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param conditional (bool, optional, default: False): being a conditional layer normalization or not.
        :param cond_hidden_units: (int, optional, default: None):
            number of units in hidden layer which projects input of conditions into `cond_hidden_units` shape.
        :param cond_hidden_activation: activation of condition hidden layer.
        :param cond_hidden_initializer: initializer of condition hidden layer.
    """

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 conditional=False,
                 cond_hidden_units=None,
                 cond_hidden_activation='linear',
                 cond_hidden_initializer='glorot_uniform',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True

        self.center = center
        self.scale = scale
        self.epsilon = epsilon or 1e-12

        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)

        self.conditional = conditional
        self.cond_hidden_units = cond_hidden_units
        self.cond_hidden_activation = keras.activations.get(cond_hidden_activation)
        self.cond_hidden_initializer = keras.initializers.get(cond_hidden_initializer)

        self.gamma, self.beta = None, None
        self.cond_hidden, self.cond_beta, self.cond_gamma = None, None, None

    def compute_output_shape(self, input_shape):
        return input_shape[0] if self.conditional else input_shape

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = [K.expand_dims(m, 0) for m in mask if m is not None]
            return None if len(masks) == 0 else K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        shape = input_shape[0][-1:] if self.conditional else input_shape[-1:]

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )

        if self.conditional:
            if self.cond_hidden_units is not None:
                self.cond_hidden = Dense(
                    units=self.cond_hidden_units,
                    kernel_initializer=self.cond_hidden_initializer,
                    activation=self.cond_hidden_activation,
                    use_bias=False,
                )
            if self.center:
                self.cond_beta = Dense(
                    units=shape[0],
                    kernel_initializer='zeros',
                    use_bias=False,
                )
            if self.scale:
                self.cond_gamma = Dense(
                    units=shape[0],
                    kernel_initializer='zeros',
                    use_bias=False,
                )

        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        beta, gamma = None, None
        if self.conditional:
            inputs, cond = inputs
            if self.cond_hidden_units is not None:
                cond = self.cond_hidden(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta + self.cond_beta(cond)
            if self.scale:
                gamma = self.gamma + self.cond_gamma(cond)
        else:
            beta = self.beta
            gamma = self.gamma

        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std

        if self.scale:
            outputs *= gamma
        if self.center:
            outputs += beta

        return outputs

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'conditional': self.conditional,
            'cond_hidden_units': self.cond_hidden_units,
            'cond_hidden_activation': keras.activations.serialize(self.cond_hidden_activation),
            'cond_hidden_initializer': keras.initializers.serialize(self.cond_hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'LayerNormalization': LayerNormalization,
}

keras.utils.get_custom_objects().update(custom_objects)
