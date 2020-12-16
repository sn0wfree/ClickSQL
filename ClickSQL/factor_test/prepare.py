# coding=utf-8
import pandas as pd
import numpy as np
import warnings


class Standardization(object):
    pass


##因子标准化
###1. 缺失值处理
##2. 去极值

def detect_series_na(series, raise_error=True):
    na_value = sum(series.isna())
    if na_value >= 1:
        msg = f'own {na_value}  NA value '
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
            return na_value
    else:
        return None


class Extrame(object):
    """
    mad
    3sigma
    percentile

    """
    __slots__ = []


if __name__ == '__main__':
    pass
