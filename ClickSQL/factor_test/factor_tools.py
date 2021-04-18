# coding=utf-8
import pandas as pd
import numpy as np
import warnings
from scipy.stats.mstats import winsorize
from scipy import linalg


def orth(df, cols=None, index=None):
    cols = df.columns if cols is None else cols
    index = df.index if index is None else index
    a2 = df.value

    orthed_a = np.array(linalg.orth(a2, rcond=0), dtype=float)
    if orthed_a.shape == a2.shape:
        return pd.DataFrame(orthed_a, index=index, columns=cols)
    else:
        raise ValueError('exists highly related factors, please check data!')


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


def _check_series_type(series):
    if isinstance(series, pd.Series):
        pass
    else:
        series = pd.Series(series)
    return series


class Outlier(object):
    """
    provide outlier detector

    """
    __slots__ = []

    @staticmethod
    def box(series, time=1.5):
        series = _check_series_type(series)
        L, U = np.percentile(sorted(series), [25, 75])
        IQR = U - L
        return np.clip(series, L - time * IQR, U + time * IQR)

    @staticmethod
    def winsorize(series, limits=[0.1, 0.2]):
        series = _check_series_type(series)
        return pd.Series(winsorize(series, limits=limits))

    @staticmethod
    def percentile(series, threshold=0.95):
        """

        :param series:
        :param threshold:
        :return:
        """
        series = _check_series_type(series)
        diff = threshold / 2.0
        lower, upper = np.percentile(sorted(series), [diff * 100, 100 - diff * 100])
        return np.clip(series, lower, upper)

    @staticmethod
    def mad(series, time=3):
        """
        median absolute deviation

        :param series:
        :param time:
        :return:
        """
        # if isinstance(series, pd.Series):
        #     pass
        # else:
        #     series = pd.Series(series)
        series = _check_series_type(series)
        median = series.quantile(0.5)
        new_median = (series - median).abs().quantile(0.5)
        upper = median + new_median * time
        lower = median - new_median * time
        return np.clip(series, lower, upper)

    @staticmethod
    def sigma(series, time=3):
        # if isinstance(series, pd.Series):
        #     pass
        # else:
        #     series = pd.Series(series)
        series = _check_series_type(series)
        std = series.std()
        mean = series.mean()

        upper = mean + std * time
        lower = mean - std * time
        return np.clip(series, lower, upper)


class Scaling(object):
    """
    squered_transform
    equal
    sqrtroot
    log
    recipocal
    boxcox
    signlog
    sigmoid

    """
    __slots__ = []

    @staticmethod
    def sigmoid(series):
        series = _check_series_type(series)
        return 1 / (1 + np.exp(-1 * series))

    @staticmethod
    def signlog(series):
        series = _check_series_type(series)
        return np.sign(series) * np.log(np.abs(series))

    @staticmethod
    def fractional(series):
        series = _check_series_type(series)
        return series / 10 ** np.ceil(np.log10(series.abs().max()))

    @staticmethod
    def range(series, range_min=0, range_max=1):
        series = _check_series_type(series)
        if range_max >= range_min:
            max = np.max(series)
            min = np.min(series)
            max_min = max - min
            if max_min > 0:
                return (series - min) / max_min * (range_max - range_min) + range_min
            else:
                raise ValueError('series got equally data or nan value')
        else:
            raise ValueError('range_max should be greater than range_min')

    @staticmethod
    def z_score(series):
        """

        :param series:
        :return:
        """
        series = _check_series_type(series)
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std

    @staticmethod
    def rank_z_score(series):
        """

        :param series:
        :return:
        """
        series = _check_series_type(series).rank()
        mean = np.mean(series)
        std = np.std(series)
        return (series - mean) / std

    @staticmethod
    def max_diff(series, ):
        """
        极差=最大值-最小值
        :param series:
        :return:
        """
        series = _check_series_type(series)
        dmax, dmin = np.max(series), np.min(series)
        max_diff = dmax - dmin
        return (series - dmin) / max_diff

    @staticmethod
    def cauterise(series):
        return series - series.mean()


if __name__ == '__main__':
    c = Scaling.signlog(np.random.random(size=(100,)))
    print(1)
    pass
