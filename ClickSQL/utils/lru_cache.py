# coding=utf-8
import functools
import time


def lru_cache(timeout: int, *lru_args, **lru_kwargs):
    def wrapper_cache(func):
        func = functools.lru_cache(*lru_args, **lru_kwargs)(func)
        func.delta = timeout
        func.expiration = time.monotonic() + func.delta

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.monotonic() >= func.expiration:
                func.cache_clear()
                func.expiration = time.monotonic() + func.delta
            return func(*args, **kwargs)

        wrapped_func.cache_info = func.cache_info
        wrapped_func.cache_clear = func.cache_clear
        return wrapped_func

    return wrapper_cache


if __name__ == '__main__':
    pass
