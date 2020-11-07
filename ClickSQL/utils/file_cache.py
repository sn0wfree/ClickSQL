# coding=utf-8
import datetime
import hashlib
import os
import pickle
from collections import OrderedDict
from functools import wraps

__refresh__ = False
DEFAULT = './'
format_dict = {'Y': '%Y', 'm': "%Y-%m", 'd': "%Y-%m-%d",
               'H': '%Y-%m-%d %H', 'M': '%Y-%m-%d %H:%M', 'S': '%Y-%m-%d %H:%M:%S'}


def get_cache_path(enable_cache: bool = False):
    dt = datetime.datetime.today().strftime('%Y%m%d')
    # __cache_path__ == f"{default}"
    cache_path = os.path.join(DEFAULT, dt)
    if not os.path.exists(cache_path) and enable_cache:
        os.mkdir(cache_path)
    return cache_path


def date_format(granularity: str):
    if granularity in format_dict.keys():
        return format_dict.get(granularity)
    else:
        raise ValueError(f'date_format not support: {granularity}')
    pass


def prepare_args(func, arg, kwargs: dict, granularity: str = 'H', exploit_func_name: bool = True,
                 enable_cache: bool = False):
    """

    :param func:
    :param arg:
    :param kwargs:
    :param granularity:
    :param exploit_func_name:  cache file name whether include func name
    :param enable_cache:
    :return:
    """
    time_format_dimension = date_format(granularity)
    dt_str = datetime.datetime.now().strftime(time_format_dimension)
    kwargs = OrderedDict(sorted(kwargs.items(), key=lambda t: t[0]))  # sort kwargs to fix hashcode if sample input
    func_name = func.__name__.__str__()
    cls_obj = func.__qualname__ != func_name
    if cls_obj:
        obj = arg[0]
        obj = obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__
        arg_tuple = tuple([obj] + list(map(str, arg[1:])))
    else:
        arg_tuple = arg
    key = pickle.dumps([func_name, arg_tuple, kwargs, dt_str])  # get the unique key for the same input
    if exploit_func_name:
        name = f"{func_name}_{hashlib.sha1(key).hexdigest()}_{dt_str}"  # create cache file name
    else:
        name = hashlib.sha1(key).hexdigest()  # create cache file name
    file_path = get_cache_path(enable_cache=enable_cache)
    return file_path, name


def write(fg, res):
    with open(fg, 'wb') as f:
        pickle.dump(res, f)


def read(fg):
    with open(fg, 'rb') as f:
        res = pickle.load(f)
    return res


def _cache(func, arg, kwargs, granularity='H', enable_cache: bool = False, exploit_func=True):
    """

    :param func:
    :param arg:
    :param kwargs:
    :param granularity:
    :param enable_cache:
    :param exploit_func:   cache file name whether include func name
    :return:
    """
    file_path, name = prepare_args(func, arg, kwargs, granularity=granularity, exploit_func_name=exploit_func,
                                   enable_cache=enable_cache)
    fg = os.path.join(file_path, name)
    if os.path.exists(fg) and enable_cache:
        return read(fg)
    else:
        res = func(*arg, **kwargs)
        if enable_cache:
            write(fg, res)
        return res


def file_cache(**deco_arg_dict):
    # if callable(deco_arg_dict):
    #     @wraps(deco_arg_dict)
    #     def wrapped(*args, **kwargs):
    #         return _cache(deco_arg_dict, args, kwargs, granularity='d', enable_cache=False)
    #
    #     return wrapped
    # else:
    def _deco(func):
        @wraps(func)
        def __deco(*args, **kwargs):
            return _cache(func, args, kwargs, **deco_arg_dict)

        return __deco

    return _deco


if __name__ == '__main__':
    @file_cache(enable_cache=True)
    def test(a, b=2):
        return a, b


    class YGH(object):
        @staticmethod
        @file_cache(enable_cache=True)
        def test(a, b=2):
            return a, b

        @classmethod
        @file_cache(enable_cache=True)
        def test2(cls, a, b=3):
            return a, b

        @file_cache(enable_cache=True)
        def test3(self, a, b=3):
            return a, b


    print(test(1, b=3))

    print(YGH.test(1, b=3))
    print(YGH.test2(1, b=3))
    print(YGH().test3(1, b=3))
    pass
