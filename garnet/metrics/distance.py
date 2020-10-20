# coding: utf-8

"""
@File   : distance.py
@Author : garnet
@Time   : 2020/10/20 23:36
"""

import numpy as np
from scipy.optimize import linprog


def wasserstein_distance(p, q, D):
    """
    Use linear programming method to calculate Wasserstein distance.
    `p` and `q` are discrete distribute, `D_{ij}` is the cost of moving from `i` to `j`.

    Shape of `p`, `q`, `D`
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    With linear limits:
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]

    See more: [从EMD、WMD到WRD：文本向量序列的相似度计算](https://kexue.fm/archives/7388)
    """

    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun
