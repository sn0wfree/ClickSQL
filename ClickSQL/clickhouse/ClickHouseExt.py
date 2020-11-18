# coding=utf-8
import pandas as pd
import re
from collections import namedtuple, ChainMap
from ClickSQL.clickhouse.ClickHouseCreate import TableEngineCreator

from ClickSQL.errors import ClickHouseTableExistsError, ParameterTypeError

factor_parameters = ('dt', 'code', 'value', 'fid')
ft_node = namedtuple('factortable', factor_parameters)


class ClickHouseTableNodeExt(TableEngineCreator):

    def __init__(self, conn_str: (str, dict, None) = None, **kwarg):
        super(ClickHouseTableNodeExt, self).__init__(conn_str=conn_str, **kwarg)
        self._src = conn_str
        self.db_table = self._para.database

    @staticmethod
    def __extend_dict_value__(conditions: (dict, ChainMap)):
        for s in conditions.values():
            if isinstance(s, str):
                yield s
            elif isinstance(s, (tuple, list)):
                for s_sub in s:
                    yield s_sub
            else:
                raise ValueError('filter settings get wrong type! only accept string and tuple of string')

    @staticmethod
    def __obtain_other_filter__(other_filters):
        exits_keys = []
        for k, v in other_filters.items():
            if k in exits_keys:
                raise ValueError(f'found duplicated key: {k}')
            exits_keys.append(k)
            if isinstance(v, dict):
                yield v
            elif isinstance(v, (str, tuple)):
                yield {k: v}
            else:
                raise ValueError('filter settings get wrong type! only accept string and tuple of string')

    @classmethod
    def _get_sql(cls, db_table: str, cols: (tuple, None, list) = None,
                 order_by_cols: (list, tuple, None) = None,
                 data_filter: dict = {}, include_filter=True,
                 **other_filters):
        """

        :param data_filter:
        :param cols:
        :param include_filter:
        :param other_filters:
        :param order_by_cols: ['test1 asc','test2 desc']
        :return:
        """
        if cols is None:
            cols = factor_parameters
        elif len(cols) == 0:
            cols = ['*']
        conditions = ChainMap(data_filter, *list(cls.__obtain_other_filter__(other_filters)))
        filter_yield = cls.__extend_dict_value__(conditions)
        if include_filter:
            cols = set(list(cols) + list(conditions.keys()))
        else:
            cols = set(cols)
        if order_by_cols is None:
            order_by_clause = ''
        elif len(order_by_cols) > 1:
            order_by_clause = f" order by ({','.join(order_by_cols)})"
        elif len(order_by_cols) == 1:
            order_by_clause = f" order by {','.join(order_by_cols)}"
        else:
            raise ValueError('order_by_cols get wrong length')
        sql = f"select {','.join(cols)} from {db_table} where {' and '.join(sorted(set(['1'] + list(filter_yield))))} {order_by_clause}"
        return sql

    def _execute(self, sql: str, **kwargs):
        return self.query(sql, **kwargs)
        # self.__execute__ = self.operator.query

    # @staticmethod
    # def _check_end_with_limit(string, pattern=r'[\s]+limit[\s]+[0-9]+$'):
    #     m = re.findall(pattern, string)
    #     if m is None or m == []:
    #         return False
    #     else:
    #         return True


if __name__ == '__main__':
    pass
