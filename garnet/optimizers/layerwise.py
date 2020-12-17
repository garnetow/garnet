# coding: utf-8

"""
@File   : layerwise.py
@Author : garnet
@Time   : 2020/12/17 16:09
"""

import keras

from ..backend import K
from ..utils.snippets import contain
from ..utils.strings import keyword_match


def extend_with_layerwise_lr(BaseOptimizer):
    r"""Inject layerwise learning rate into keras Optimizer instance.

    Learning rate of specified layers will be different from global learning rate. See more at
    https://kexue.fm/archives/6418.

    Args:
        :param BaseOptimizer: keras `Optimizer` instance
    """

    class LayerwiseLrOptimizer(BaseOptimizer):
        r"""
        Args:
            :param params_rate: `dict` of `{parameter name: scale rate}` mapping. While updating, learning rate of
                parameters with specified keyword will multiply corresponding scale rate.
        """

        def __init__(self, params_rate: dict, with_self_adaption=True, *args, **kwargs):
            super(LayerwiseLrOptimizer, self).__init__(*args, **kwargs)
            self.params_rate = params_rate
            self.with_self_adaption = with_self_adaption
            self.power = 1 if self.with_self_adaption else 2

        @K.symbolic
        def get_updates(self, loss, params):
            hold_update = K.update

            def new_update(x, x_new):
                r"""Update the value of `x` to `new_x`.

                Args:
                    x: A `Variable`.
                    new_x: A tensor of same shape as `x`.

                Returns:
                    The variable `x` updated.
                """
                if contain(x, params):
                    lr_multiplier = 1
                    for name, rate in self.params_rate.items():
                        if name in x.name:
                            lr_multiplier *= rate ** self.power
                    if lr_multiplier != 1:
                        x_new = x + (x_new - x) * lr_multiplier
                return hold_update(x, x_new)

            K.update = new_update
            updates = super(LayerwiseLrOptimizer, self).get_updates(loss, params)
            K.update = hold_update

            return updates

        def get_config(self):
            config = {
                'params_rate': self.params_rate,
                'with_self_adaption': self.with_self_adaption,
            }
            base_config = super(LayerwiseLrOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return LayerwiseLrOptimizer
