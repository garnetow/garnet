# coding: utf-8

"""
@File   : decorator.py
@Author : garnet
@Time   : 2020/12/1 11:49
"""

from functools import wraps


def safe_return(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    return wrapper
