# coding=utf-8

import numpy as np
from functools import partial
from itertools import starmap
import pandas as pd


class SingleBrinson(object):
    """
    https://blog.csdn.net/qq_43382509/article/details/106035440
    """

    @staticmethod
    def cal_single_asi(rp: pd.Series, rb: pd.Series, wp: pd.Series, wb: pd.Series):
        """
        brinson 单期模型

        :param wp: portfolio weight
        :param rp: portfolio return matrix among single asset
        :param wb: benchmark weight
        :param rb: benchmark return matrix among single asset
        :return:
        """
        Rp = np.sum(np.dot(rp, wp))
        Rb = np.sum(np.dot(rb, wb))
        Re = Rp - Rb  # RE 为总超额收益

        Ra = np.sum(np.dot((wp - wb), rb))  # RTS 为择时收益
        Rs = np.sum(np.dot(wb, (rp - rb)))  # RSS 为择股收益
        Ri = np.sum(np.dot((wp - wb), (rp - rb)))  # RIE 为交互作用
        return Re, Ra, Rs, Ri

    pass


class MultiBrinson(object):
    """
    多期Brinson业绩归因分析——Carino模型
    https://zhuanlan.zhihu.com/p/42893026?utm_source=wechat_session
    """

    @staticmethod
    def cal_k(r, b):
        if r == b:
            return (np.log(1 + r) - np.log(1 + b)) / (r - b)
        else:
            return 1 / (1 + r)

    @classmethod
    def cal_kt(cls, rt, bt):
        return cls.cal_k(rt, bt)

    @staticmethod
    def cal_single(xt, kt, k, ):
        return (kt / k) * xt

    @classmethod
    def cal_a(cls, at, kt, k):
        return cls.cal_single(kt, k, at)

    @classmethod
    def cal_s(cls, st, kt, k):
        return cls.cal_single(kt, k, st)

    @classmethod
    def cal_i(cls, it, kt, k):
        return cls.cal_single(kt, k, it)

    @classmethod
    def cal_asi_base(cls, A_series, S_series, I_series, K_series, k):
        func = partial(cls.cal_single, k=k)
        A = sum(map(func, zip(A_series, K_series)))
        S = sum(map(func, zip(S_series, K_series)))
        I = sum(map(func, zip(I_series, K_series)))
        return A, S, I

    @classmethod
    def cal_asi_raw(cls, A_series, S_series, I_series, rt, bt, r, b):
        k = cls.cal_k(r, b)
        K_series = map(cls.cal_k, zip(rt, bt))
        return cls.cal_asi_base(cls, A_series, S_series, I_series, K_series, k)

    @classmethod
    def cal_asi(cls, wp: pd.Series, rp: pd.DataFrame, wb: pd.Series, rb: pd.DataFrame):
        Rp = np.dot(rp, wp)
        Rb = np.dot(rb, wb)
        r = np.prod(Rp + 1) - 1
        b = np.prod(Rb + 1) - 1
        func = partial(SingleBrinson.cal_single_asi, wp=wp, wb=wb)
        # tasks = [(p, b) for p, b in ]
        Re, A_series, S_series, I_series = zip(*list(starmap(func, zip(Rp, Rb))))
        return cls.cal_asi_raw(A_series, S_series, I_series, Rp, Rb, r, b)


if __name__ == '__main__':
    pass
