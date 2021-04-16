# coding=utf-8
import os
import uuid

from ClickSQL.utils import singleton, uuid_hash

@singleton
class Conf(object):

    def __setitem__(self, key: str, value):
        os.environ['CONFIG:' + uuid_hash(key).upper()] = value

    def __getitem__(self, item):
        return os.getenv('CONFIG:' + uuid_hash(item).upper())

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, item):
        return self.__getitem__(self, item)

    @staticmethod
    def show_configs():
        return {k: os.getenv(k) for k in filter(lambda x: x.startswith('CONFIG:') and len(x) == 39, os.environ.keys())}


Config = Conf()

if __name__ == '__main__':
    # C = Conf()
    # C.test = '1'
    #
    # print(C.show_config())

    pass
