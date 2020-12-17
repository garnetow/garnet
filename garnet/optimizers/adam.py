# coding: utf-8

"""
@File   : adam.py
@Author : garnet
@Time   : 2020/12/17 14:55
"""

import keras

from ..backend import K
from ..utils.snippets import contain
from ..utils.strings import keyword_match


class AdamW(keras.optimizers.Adam):
    r"""Adam with weight decay
    """

    def __init__(self, weight_decay_rate=0.01, exclude_params=None, *args, **kwargs):
        super(AdamW, self).__init__(*args, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self.exclude_params = exclude_params or []

    @K.symbolic
    def get_updates(self, loss, params):
        hold_update = K.update

        def new_update(x, new_x):
            r"""Update the value of `x` to `new_x`.

            Args:
                x: A `Variable`.
                new_x: A tensor of same shape as `x`.

            Returns:
                The variable `x` updated.
            """
            if contain(x, params) and self._should_decay(x):
                new_x = new_x - self.learning_rate * self.weight_decay_rate * x
            return hold_update(x, new_x)

        K.update = new_update
        updates = super(AdamW, self).get_updates(loss, params)
        K.update = hold_update

        return updates

    def _should_decay(self, var):
        return not keyword_match(var.name, self.exclude_params)

    def get_config(self):
        config = {
            'weight_decay_rate': self.weight_decay_rate,
            'exclude_params': self.exclude_params,
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    'AdamW': AdamW,
}

keras.utils.get_custom_objects().update(custom_objects)
