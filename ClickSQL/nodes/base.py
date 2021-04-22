# coding=utf-8
import copy
import warnings
from collections import namedtuple, deque

from functools import wraps

from ClickSQL.clickhouse.ClickHouseExt import ClickHouseTableNodeExt
from ClickSQL.errors import ClickHouseTableNotExistsError
from ClickSQL.nodes.groupby import GroupSQLUtils
from ClickSQL.nodes.merge import MergeSQLUtils

complex_sql_select_count = 4
factor_parameters = ('dt', 'code', 'value', 'fid')
ft_node = namedtuple('factortable', factor_parameters)

# class FactorBackendCH(object):
#     __slots__ = ['_src', 'node', 'db_table']
#
#     def __init__(self, src: str):
#         """
#
#         :param src:  sample: clickhouse://test:sysy@199.199.199.199:1234/drre
#         """
#         self._src = src
#         self.node = ClickHouseTableNodeExt(conn_str=src)
#         self.db_table = self.node._para.database
#
#     def __call__(self, sql: str, **kwargs):
#         return self.node.query(sql, **kwargs)


"add auto-increment col by materialized bitOr(bitShiftLeft(toUInt64(now64()),24), rowNumberInAllBlocks()) "


class DelayTasks(deque):
    def __init__(self,_query, *args, **kwargs):
        super(DelayTasks, self).__init__(*args, **kwargs)

        self._query = _query

    def run(self, no_yield=False):
        result = (self._query(sql) for sql in self)
        if no_yield:
            return list(result)

        else:
            return result


class BaseSingleQueryBaseNode(object):
    """
    init operator
    and set fid_ck,dt_max_1st,execute,no_self_update for update functions
    kwargs is the conditions for select
    """
    __Name__ = "基础因子库单因子基类"
    __slots__ = (
        'operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', 'status', '_INFO', 'depend_tables',
        '_fid_ck', '_dt_max_1st', '_execute', '_no_self_update', 'delay_tasks'
    )

    def __init__(self, src: str, db_table: (None, str) = None, info=None, **kwargs):
        """

        :type kwargs: object
        :param src: string sample: clickhouse://test:sysy@199.199.199.199:1234/drre
        :param db_table:
        :param info:
        :param kwargs:  data_filter will store operator for some cols:
                { cols: (tuple, None, list) = None,
                 order_by_cols: (list, tuple, None) = None,
                 data_filter: dict = {}, include_filter=True,
                 **other_filters}




        """

        self.operator = ClickHouseTableNodeExt(src)

        # self._execute = self._operator._execute

        if db_table is None:
            src_db_table = self.operator.db_table
            db_table_all, db, table = self._db_split(src_db_table)
            # if '.' in src_db_table:
            #     self.db_table = src_db_table
            # else:
            #     raise ValueError('db_table parameter get wrong type!')
        elif isinstance(db_table, str):
            db_table_all, db, table = self._db_split(db_table)
        else:
            raise ValueError('db_table only accept str!')
        self.db_table = db_table_all
        # db, table = self.db_table.split('.')
        self.db = db
        self.table = table
        # self.depend_tables = [self.db_table]
        self._kwargs = kwargs
        self._raw_kwargs = kwargs
        self.status = 'SQL'
        self._INFO = info

        self.delay_tasks = DelayTasks(self.operator)
        # self.delay_tasks._query = self.operator

    @staticmethod
    def _db_split(src_db_table):
        if isinstance(src_db_table, str):
            if '.' in src_db_table:
                db, table = src_db_table.split('.')
                return src_db_table, db, table
            else:
                return f"{src_db_table}.None", src_db_table, 'None'
        else:
            raise ValueError('db_table parameter get wrong type! only accept str')

    @wraps(ClickHouseTableNodeExt.insert_df)
    def insert_df(self, *args, **kwargs):
        self.operator.insert_df(*args, **kwargs)

    # create table
    @wraps(ClickHouseTableNodeExt.create)
    def create(self, *args, **kwargs):
        return self.operator.create(*args, **kwargs)

    def _update(self, **kwargs):
        """
        update kwargs settings
        :param kwargs:
        :return:
        """

        self._kwargs.update(kwargs)

    def __str__(self, *args, **kwargs):
        return self.__sql__

    def __len__(self) -> int:
        """
        Returns length of info axis, but here we use the index.
        """
        sql = f"select count(1) as rows from ({self.__sql__})"
        rows = self.operator(sql)['rows'].values[0]
        return rows

    @property
    def __sql__(self):
        return self.operator.get_sql(db_table=self.db_table, **self._kwargs)

    def decorate(self, target_conn_args='conn'):
        def afunc(func):
            def _afunc(*args, **kwargs):
                if target_conn_args in kwargs.keys():
                    if kwargs[target_conn_args] is None:
                        kwargs[target_conn_args] = self.operator
                    elif callable(kwargs[target_conn_args]):
                        pass
                    else:
                        raise KeyError(f'target variable:{target_conn_args} had been setup into other value! ')
                else:
                    raise KeyError(f'cannot find target variable:{target_conn_args} ')
                return func(*args, **kwargs)

            return _afunc

        return afunc

    def __call__(self, *sql, **kwargs):
        """

        add delay function to delay execute sql

        :param sql:
        :param kwargs:
        :return:
        """

        if 'delay' in kwargs.keys():
            if kwargs['delay']:
                if isinstance(sql, str):
                    self.delay_tasks.append(sql)
                else:
                    for s in sql:
                        self.delay_tasks.append(s)
            else:
                kwargs.pop('delay')
                return self.operator(*sql, **kwargs)
        else:

            return self.operator(*sql, **kwargs)

    @property
    def __factor_id__(self):  # add iid function get factor table id
        return hash(self.__sql__)

    def __getitem__(self, key: (list, str)):
        """
        directly execute
        :param key:
        :return:
        """
        if isinstance(key, list):
            sql = f"select {','.join(key)} from ({self.__sql__})"
            return self.operator(sql)
        elif isinstance(key, str):
            sql = f"select {key} from ({self.__sql__})"
            return self.operator(sql)
        else:
            raise ValueError('key only accept list or str')

    @property
    def __system_tables__(self):
        sql = f"select total_rows,engine from system.tables where database ='{self.db}' and name='{self.table}'"
        res = self.operator(sql)
        return res

    def _detect_complex_sql(self, warn=False):
        """
        detect complex sql
        :param warn:
        :return:
        """
        sql = self.__sql__.lower()
        if 'select' in sql and len(sql.split('select ')) >= complex_sql_select_count:
            if warn:
                warnings.warn('this sql may be a complex sql!')
            else:
                raise ValueError('this sql may be a complex sql!')

    @property
    def table_exist(self):
        """
        return table exists status
        :return:
        """

        return not self.__system_tables__.empty

    @property
    def table_engine(self):
        """
        return table engine
        :return:
        """
        if self.table_exist:
            return self.__system_tables__['engine'].values[0]
        else:
            raise ClickHouseTableNotExistsError(f'{self.db_table} is not exists!')

    def fetch(self, num=1000, pattern=r'[\s]+limit[\s]+[0-9]+$', **kwargs):
        """
        fetch first 1000 line
        :return:
        """
        sql = self.__sql__
        self._detect_complex_sql()
        end_with_limit = self.operator._check_end_with_limit(sql, pattern=pattern)
        if end_with_limit:
            return self.operator(sql, **kwargs)
        else:
            try:
                return self.operator(sql + f' limit {num}', **kwargs)
            except Exception as e:
                return self.operator(sql, **kwargs)

    def fetch_all(self, **kwargs):
        """
        fetch all data
        :return:
        """
        self._detect_complex_sql()
        return self.operator(self.__sql__, **kwargs)

    @property
    def row_count(self):
        return self.__len__()

    @property
    def shape(self):
        """
        shape like pandas

        :return:
        """

        return self.row_count, self.col_count

    @property
    def dtypes(self):
        sql = f"desc ({self.__sql__})"
        dtypes = self.operator(sql)
        return dtypes

    @property
    def columns(self):
        return self.dtypes['name'].values.tolist()

    @property
    def col_count(self):
        return self.dtypes.shape[0]

    @property
    def empty(self):
        return self.total_rows == 0

    @property
    def total_rows(self):
        """
        return row count
        :return:
        """

        # sql = f"-- select count(1) as row_count from {self.db_table}"
        temp = self.__system_tables__
        if not temp.empty:
            return temp['total_rows'].values[0]
        else:
            raise ClickHouseTableNotExistsError(f'{self.db_table} is not exists!')

    def nlargest(self, top: int, columns: list, execute: bool = True, extra_cols: (str, None) = None):
        """
        return largest n by columns
        :param extra_cols:
        :param top:
        :param columns:
        :param execute:
        :return:
        """
        by = [f'{c} desc' for c in columns]
        sql = GroupSQLUtils.group_top(self.__sql__, by=by, top=top, cols=extra_cols)
        if execute:
            return self.operator(sql)
        else:
            return sql

    def nsmallest(self, top: int, columns: list, execute: bool = True, extra_cols: (str, None) = None):
        """
        return largest n by columns
        :param extra_cols:
        :param top:
        :param columns:
        :param execute:
        :return:
        """
        by = [f'{c} asc' for c in columns]
        sql = GroupSQLUtils.group_top(self.__sql__, by=by, top=top, cols=extra_cols)
        if execute:
            return self.operator(sql)
        else:
            return sql

    def groupby(self, by: (str, list, tuple), apply_func: (list,), having: (list, tuple, None) = None, execute=True):
        """

        :param execute:
        :param by:
        :param apply_func:
        :param having:
        :return:
        """
        sql = GroupSQLUtils.group_by(self.__sql__, by=by, apply_func=apply_func, having=having)
        if execute:
            return self.operator(sql)
        else:
            return sql

    def merge(self, seconds, using: (list, str, tuple), join_type='all full join', cols: (list, str, None) = None,
              execute=True):
        """

        :param seconds:
        :param using:
        :param join_type:
        :param cols:
        :param execute:
        :return:
        """
        if seconds.lower().startswith('select'):
            pass
        else:
            seconds = f" select * from {seconds}"
        sql = MergeSQLUtils._merge(self, seconds, using=using, join_type=join_type, cols=cols)
        if execute:
            return self.operator(sql)
        else:
            return sql

    # update table
    # def __lshift__(self, src_db_table):
    #     print('lshift')
    #     fid_ck = self._fid_ck
    #     dt_max_1st = self._dt_max_1st
    #     execute = self._execute
    #     no_self_update = self._no_self_update
    #
    #     if isinstance(src_db_table, str):
    #         src_conn = copy.deepcopy(self.operator._src).replace(self.db_table, src_db_table)
    #         src_db_table = BaseSingleQueryBaseNode(src_conn, cols=['*'])
    #     elif isinstance(src_db_table, BaseSingleQueryBaseNode):
    #         pass
    #     else:
    #         raise ValueError('src_db_table is not valid! please check!')
    #
    #     if src_db_table.empty:
    #         raise ValueError(f'{src_db_table.db_table} is empty')
    #     # check two table are same
    #     if no_self_update and self.db_table == src_db_table.db_table and self.__factor_id__ == src_db_table.__factor_id__:
    #         dst = src_db_table.db_table
    #         src = self.db_table
    #         raise ValueError(
    #             f'Detect self-update process! these operator attempts to update data from {src} to {dst}')
    #
    #     update_status = 'full' if self.empty else 'incremental'
    #
    #     func = getattr(UpdateSQLUtils, f'{update_status}_update')
    #     sql = func(src_db_table, self.db_table, fid_ck, dt_max_1st=dt_max_1st)
    #     if execute:
    #         self.operator(sql)
    #     return sql, update_status
    #
    # # update table
    # def __rshift__(self, dst_db_table: str):
    #     """
    #
    #     UpdateSQLUtils
    #
    #     :param dst_db_table:
    #     :return:
    #     """
    #     # print('rshift')
    #     fid_ck = self._fid_ck
    #     dt_max_1st = self._dt_max_1st
    #     execute = self._execute
    #     no_self_update = self._no_self_update
    #     if self.empty:
    #         raise ValueError(f'{self.db_table} is empty')
    #
    #     if isinstance(dst_db_table, str):
    #         dst_conn = copy.deepcopy(self.operator._src)
    #         dst_db_table = BaseSingleQueryBaseNode(
    #             dst_conn.replace(self.db_table, dst_db_table),
    #             cols=['*']
    #         )
    #     elif isinstance(dst_db_table, BaseSingleQueryBaseNode):
    #         pass
    #     else:
    #         raise ValueError('dst_db_table is not valid! please check!')
    #     # check two table are same
    #     if no_self_update and self.db_table == dst_db_table.db_table:
    #         if self.__factor_id__ == dst_db_table.__factor_id__:
    #             dst = dst_db_table.db_table
    #             src = self.db_table
    #             raise ValueError(
    #                 f'Detect self-update process! these operator attempts to update data from {src} to {dst}')
    #
    #     update_status = 'full' if dst_db_table.empty else 'incremental'
    #
    #     func = getattr(UpdateSQLUtils, f'{update_status}_update')
    #     sql = func(self, dst_db_table, fid_ck, dt_max_1st=dt_max_1st)
    #     if execute:
    #         self.operator(sql)
    #     return sql, update_status


# class BaseSingleFactorNode(BaseSingleFactorBaseNode):
#     __slots__ = (
#         'operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', 'status', '_INFO', 'depend_tables',
#         '_fid_ck', '_dt_max_1st', '_execute', '_no_self_update'
#     )
#
#     def __init__(self, *args, **kwargs):
#         super(BaseSingleFactorNode, self).__init__(*args, **kwargs)


# class UpdateSQLUtils(object):
#
#     @staticmethod
#     def full_update(src_db_table: BaseSingleQueryBaseNode, dst_db_table: BaseSingleQueryBaseNode, **kwargs):
#         # dst_db_table = dst_db_table.db_table
#         # dst_db, dst_table = dst_db_table.db, dst_db_table.table
#         dst_table_type = dst_db_table.table_engine
#         dst = dst_db_table.db_table
#         if dst_table_type == 'View':
#             raise ValueError(f'{dst} is View ! cannot be updated!')
#         insert_sql = f"insert into {dst} {src_db_table}"
#         return insert_sql
#
#     @staticmethod
#     def incremental_update(src_db_table: BaseSingleQueryBaseNode, dst_db_table: BaseSingleQueryBaseNode,
#                            fid_ck: str, dt_max_1st=True, inplace=False, **kwargs):
#         # src_db_table = src_table.db_table
#         # src_table_type = src_db_table.table_engine
#         dst_table_type = dst_db_table.table_engine
#         dst = dst_db_table.db_table
#         if dst_table_type == 'View':
#             raise ValueError(f'{dst} is View ! cannot be updated!')
#         if dt_max_1st:
#             order_asc = ' desc'
#         else:
#             order_asc = ' asc'
#         sql = f" select distinct {fid_ck} from {dst} order by {fid_ck} {order_asc} limit 1 "
#         fid_ck_values = src_db_table.operator(sql).values.ravel().tolist()[0]
#         if inplace:
#             src_db_table._update(**{f'{fid_ck} as src_{fid_ck}': f' {fid_ck} > {fid_ck_values}'})
#             insert_sql = f"insert into {dst} {src_db_table}"
#         else:
#             src_db_table_copy = copy.deepcopy(src_db_table)
#             src_db_table_copy._update(**{f'{fid_ck} as src_{fid_ck}': f' {fid_ck} > {fid_ck_values}'})
#             insert_sql = f"insert into {dst} {src_db_table_copy}"
#
#         return insert_sql


class BaseSingleFactorTableNode(BaseSingleQueryBaseNode):
    __slots__ = (
        'operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', 'status', '_INFO', 'depend_tables',
        # '_fid_ck', '_dt_max_1st',
        '_execute', '_no_self_update', '_complex'
    )

    def __init__(self, src: str, db_table: (None, str) = None, info=None,
                 # fid_ck: str = 'fid', dt_max_1st: bool = True,
                 execute: bool = False,
                 # no_self_update: bool = True,
                 **kwargs):
        super(BaseSingleFactorTableNode, self).__init__(src, db_table=db_table, info=info, **kwargs)
        self._complex = False
        # self._fid_ck = fid_ck
        # self._dt_max_1st = dt_max_1st
        self._execute = execute
        # self._no_self_update = no_self_update

    # def execute(self, sql):
    #     ## add execute usage
    #     return self.operator(sql)
    #
    # def drop_table(self, target: str):
    #     if '.' not in target:
    #         raise ValueError('drop table must tell correspond database')
    #     self.operator(f'drop table if exists {target}')
    #
    # def drop_db(self, target: str):
    #     self.operator(f'drop database if exists {target}')


## https://zhuanlan.zhihu.com/p/297623539


if __name__ == '__main__':
    # v_st_dis_buy_info = BaseSingleFactorTableNode(
    #     'clickhouse://default:Imsn0wfree@47.104.186.157:8123/raw.v_st_dis_buy_info',
    #     cols=None,
    #     order_by_cols=['money asc'],
    #     money='money >= 1000000', limit='limit 10'
    # )
    # print(v_st_dis_buy_info)
    #
    # # factor >> 'test.test'
    # # print(factor)
    # c = v_st_dis_buy_info.fetch(1000)
    # print(c)
    # sql = v_st_dis_buy_info.merge('select cust_no, product_id, money as s from sample.sample where money >= 100000', using=['cust_no', 'product_id'])
    # print(sql)
    # c = factor('show tables from raw')
    # c2 = factor.groupby(['test2'], apply_func=['sum(fid)'])
    # print(c2)

    # print(1 >> 2)
    import numpy as np
    import pandas as pd

    data = np.random.random(size=(100, 2))
    res = pd.DataFrame(data, columns=['test1', 'test2'])
    res.groupby('test1')

    pass
