# coding: utf-8

"""
@File   : snippets.py
@Author : garnet
@Time   : 2020/12/17 14:35
"""


def contain(x, ys):
    r"""Equals to `x in ys`
    """
    for y in ys:
        if x is y:
            return True
    return False
