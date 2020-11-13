# coding=utf-8
import pandas as pd
import re

from ClickSQL.clickhouse.ClickHouseCreate import TableEngineCreator

from ClickSQL.errors import ClickHouseTableExistsError, ParameterTypeError


class ClickHouseTableNodeExt(TableEngineCreator):
    def __init__(self, conn_str: (str, dict, None) = None, **kwarg):
        super(ClickHouseTableNodeExt, self).__init__(conn_str=conn_str, **kwarg)


if __name__ == '__main__':
    pass
