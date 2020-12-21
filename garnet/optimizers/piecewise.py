# coding: utf-8

"""
@File   : piecewise.py
@Author : garnet
@Time   : 2020/12/21 11:42
"""

from ..backend import K
from ..utils.snippets import contain


def extend_with_piecewise_lr(BaseOptimizer):
    r"""Inject piecewise learning rate into keras Optimizer instance.

    During training process, learning rate piecewise changes according to step-rate mapping schedule.
    For example, given schedule `{1000: 1, 2000: 0.1}`, learning rate will change from 0% to 100% in 0~1000 steps,
    and then change from 100% to 10% in 1001~2000 steps. After 2000th step, learning rate will keep 10%.

    Default initial learning rate will be 100%, and `{0: 0.01}` mean initial learning rate is 10%.

    Args:
        :param BaseOptimizer: keras `Optimizer` instance
    """

    class PiecewiseLrOptimizer(BaseOptimizer):
        r"""
        Args:
            :param piecewise_rate: `dict` of `{parameter name: scale rate}` mapping. While updating, learning rate of
                parameters with specified keyword will multiply corresponding scale rate.
        """

        def __init__(self, lr_schedule: {0: 1}, *args, **kwargs):
            super(PiecewiseLrOptimizer, self).__init__(*args, **kwargs)

            schedule = sorted(lr_schedule.items())
            if len(schedule) == 0 or schedule[0][0] != 0:
                schedule = [(0, 0.0)] + schedule
            self.schedule = schedule

        def piecewise_linear(self, epoch):
            x = K.constant(self.schedule[0][1], dtype=K.floatx())
            t = K.cast(epoch, K.floatx())

            for i in range(len(self.schedule)):
                t_begin = self.schedule[i][0]
                x_begin = x
                if i != len(self.schedule) - 1:
                    dx = self.schedule[i + 1][1] - self.schedule[i][1]
                    dt = self.schedule[i + 1][0] - self.schedule[i][0]
                    slope = 1.0 * dx / dt
                    x = self.schedule[i][1] + slope * (t - t_begin)
                else:
                    x = K.constant(self.schedule[i][1], dtype=K.floatx())
                x = K.switch(t >= t_begin, x, x_begin)
            return x

        @K.symbolic
        def get_updates(self, loss, params):
            lr_multiplier = self.piecewise_linear(self.iterations)
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
                    x_new = x + (x_new - x) * lr_multiplier
                return hold_update(x, x_new)

            K.update = new_update
            updates = super(PiecewiseLrOptimizer, self).get_updates(loss, params)
            K.update = hold_update

            return updates

        def get_config(self):
            config = {
                'schedule': self.schedule,
            }
            base_config = super(PiecewiseLrOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return PiecewiseLrOptimizer
