# coding=utf-8
import os
import json
from ClickSQL.utils import singleton, uuid_hash


@singleton
class Conf(object):

    def __setitem__(self, key: str, value):
        os.environ['CONFIG:' + uuid_hash(key).upper()] = json.dumps(value)

    def __getitem__(self, item: str):
        result = os.getenv('CONFIG:' + uuid_hash(item).upper())
        if result is None:
            return result
        else:
            return json.loads(result)

    @staticmethod
    def get(key: str, default=None):
        """

        :param key:
        :param default:
        :return:
        """

        key_str = 'CONFIG:' + uuid_hash(key).upper()
        result = os.environ.get(key_str, default=default)
        if result is None:
            return None
        else:

            return json.loads(result)

    @staticmethod
    def set(key: str, value):

        key_str = 'CONFIG:' + uuid_hash(key).upper()

        os.environ[key_str] = json.dumps(value)

    def __setattr__(self, key: str, value: str):
        self.__setitem__(key, value)

    def __getattr__(self, item: str):
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
