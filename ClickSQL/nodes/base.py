# coding=utf-8
import copy
import warnings
from collections import namedtuple, deque, Callable

import pandas as pd
import time
from functools import wraps

from ClickSQL.clickhouse.ClickHouseExt import ClickHouseTableNodeExt
from ClickSQL.errors import ClickHouseTableNotExistsError
from ClickSQL.nodes.groupby import GroupSQLUtils
from ClickSQL.nodes.merge import MergeSQLUtils

complex_sql_select_count = 4
factor_parameters = ('dt', 'code', 'value', 'fid')
ft_node = namedtuple('factortable', factor_parameters)

CIK = namedtuple('CoreIndexKeys', ('dts', 'iid'))
CIKDATA = namedtuple('CoreIndexKeys', ('dts', 'iid'))
FactorInfo = namedtuple('FactorInfo', ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions'))

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


class BaseSingleQueryBaseNode(object):
    """
    init operator
    and set fid_ck,dt_max_1st,execute,no_self_update for update functions
    kwargs is the conditions for select
    """
    __Name__ = "基础因子库单因子基类"
    __slots__ = (
        'operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', 'status', '_INFO', 'depend_tables',
        '_fid_ck', '_dt_max_1st', '_execute', '_no_self_update'
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
            db_table_all, db, table = self.db_split(src_db_table)
            # if '.' in src_db_table:
            #     self.db_table = src_db_table
            # else:
            #     raise ValueError('db_table parameter get wrong type!')
        elif isinstance(db_table, str):
            db_table_all, db, table = self.db_split(db_table)
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

    @staticmethod
    def db_split(src_db_table):
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
    def __lshift__(self, src_db_table):
        print('lshift')
        fid_ck = self._fid_ck
        dt_max_1st = self._dt_max_1st
        execute = self._execute
        no_self_update = self._no_self_update

        if isinstance(src_db_table, str):
            src_conn = copy.deepcopy(self.operator._src).replace(self.db_table, src_db_table)
            src_db_table = BaseSingleQueryBaseNode(src_conn, cols=['*'])
        elif isinstance(src_db_table, BaseSingleQueryBaseNode):
            pass
        else:
            raise ValueError('src_db_table is not valid! please check!')

        if src_db_table.empty:
            raise ValueError(f'{src_db_table.db_table} is empty')
        # check two table are same
        if no_self_update and self.db_table == src_db_table.db_table and self.__factor_id__ == src_db_table.__factor_id__:
            dst = src_db_table.db_table
            src = self.db_table
            raise ValueError(
                f'Detect self-update process! these operator attempts to update data from {src} to {dst}')

        update_status = 'full' if self.empty else 'incremental'

        func = getattr(UpdateSQLUtils, f'{update_status}_update')
        sql = func(src_db_table, self.db_table, fid_ck, dt_max_1st=dt_max_1st)
        if execute:
            self.operator(sql)
        return sql, update_status

    # update table
    def __rshift__(self, dst_db_table: str):
        """

        UpdateSQLUtils

        :param dst_db_table:
        :return:
        """
        # print('rshift')
        fid_ck = self._fid_ck
        dt_max_1st = self._dt_max_1st
        execute = self._execute
        no_self_update = self._no_self_update
        if self.empty:
            raise ValueError(f'{self.db_table} is empty')

        if isinstance(dst_db_table, str):
            dst_conn = copy.deepcopy(self.operator._src)
            dst_db_table = BaseSingleQueryBaseNode(
                dst_conn.replace(self.db_table, dst_db_table),
                cols=['*']
            )
        elif isinstance(dst_db_table, BaseSingleQueryBaseNode):
            pass
        else:
            raise ValueError('dst_db_table is not valid! please check!')
        # check two table are same
        if no_self_update and self.db_table == dst_db_table.db_table:
            if self.__factor_id__ == dst_db_table.__factor_id__:
                dst = dst_db_table.db_table
                src = self.db_table
                raise ValueError(
                    f'Detect self-update process! these operator attempts to update data from {src} to {dst}')

        update_status = 'full' if dst_db_table.empty else 'incremental'

        func = getattr(UpdateSQLUtils, f'{update_status}_update')
        sql = func(self, dst_db_table, fid_ck, dt_max_1st=dt_max_1st)
        if execute:
            self.operator(sql)
        return sql, update_status

    def execute(self, sql):
        ## add execute usage
        return self.operator(sql)

    def drop_table(self, target: str):
        if '.' not in target:
            raise ValueError('drop table must tell correspond database')
        self.operator(f'drop table if exists {target}')

    def drop_db(self, target: str):
        self.operator(f'drop database if exists {target}')


class UpdateSQLUtils(object):

    @staticmethod
    def full_update(src_db_table: BaseSingleQueryBaseNode, dst_db_table: BaseSingleQueryBaseNode, **kwargs):
        # dst_db_table = dst_db_table.db_table
        # dst_db, dst_table = dst_db_table.db, dst_db_table.table
        dst_table_type = dst_db_table.table_engine
        dst = dst_db_table.db_table
        if dst_table_type == 'View':
            raise ValueError(f'{dst} is View ! cannot be updated!')
        insert_sql = f"insert into {dst} {src_db_table}"
        return insert_sql

    @staticmethod
    def incremental_update(src_db_table: BaseSingleQueryBaseNode, dst_db_table: BaseSingleQueryBaseNode,
                           fid_ck: str, dt_max_1st=True, inplace=False, **kwargs):
        # src_db_table = src_table.db_table
        # src_table_type = src_db_table.table_engine
        dst_table_type = dst_db_table.table_engine
        dst = dst_db_table.db_table
        if dst_table_type == 'View':
            raise ValueError(f'{dst} is View ! cannot be updated!')
        if dt_max_1st:
            order_asc = ' desc'
        else:
            order_asc = ' asc'
        sql = f" select distinct {fid_ck} from {dst} order by {fid_ck} {order_asc} limit 1 "
        fid_ck_values = src_db_table.operator(sql).values.ravel().tolist()[0]
        if inplace:
            src_db_table._update(**{f'{fid_ck} as src_{fid_ck}': f' {fid_ck} > {fid_ck_values}'})
            insert_sql = f"insert into {dst} {src_db_table}"
        else:
            src_db_table_copy = copy.deepcopy(src_db_table)
            src_db_table_copy._update(**{f'{fid_ck} as src_{fid_ck}': f' {fid_ck} > {fid_ck_values}'})
            insert_sql = f"insert into {dst} {src_db_table_copy}"

        return insert_sql


def SmartDataFrame(df: pd.DataFrame, db_table: str, dts: str, iid: str, origin_factor_names: str, alias: str, sql: str,
                   conditions: str):
    sproperty = {'_db_table': property(lambda x: db_table),
                 '_cik_dt': property(lambda x: dts),
                 '_cik_iid': property(lambda x: iid),
                 '_origin_factor_names': property(lambda x: origin_factor_names),
                 '_alias': property(lambda x: alias),
                 '_sql': property(lambda x: sql),
                 '_conditions': property(lambda x: conditions),

                 }

    result_cls = type('SmartDataFrame', (pd.DataFrame,), sproperty)
    return result_cls(df)


class FactorCheckHelper(object):
    @staticmethod
    def check_alias(factor_names: (list,), as_alias: (list, tuple, str) = None):
        if as_alias is None:
            alias = len(factor_names) * [None]
        elif isinstance(as_alias, str):
            alias = [as_alias]
        elif isinstance(as_alias, (list, tuple)):
            if len(as_alias) != len(factor_names):
                raise ValueError('as_alias is not match factor_names')
            else:
                alias = as_alias
        else:
            raise ValueError('alias only accept list tuple str!')
        return alias

    @staticmethod
    def check_factor_names(factor_names: (list, tuple, str)):
        if isinstance(factor_names, str):
            factor_names = [factor_names]
        elif isinstance(factor_names, (list, tuple)):
            factor_names = list(factor_names)
        else:
            raise ValueError('columns only accept list tuple str!')
        return factor_names

    @staticmethod
    def check_cik_dt(cik_dt, default_cik_dt):
        if cik_dt is not None:
            pass

        elif cik_dt is None:
            cik_dt = default_cik_dt
        else:
            raise NotImplementedError('cik_dt is not setup!')
        return cik_dt

    @staticmethod
    def check_cik_iid(cik_iid, default_cik_iid):
        if cik_iid is not None:
            pass

        elif cik_iid is None:
            cik_iid = default_cik_iid
        else:
            raise NotImplementedError('cik_dt is not setup!')
        return cik_iid


class _Factors(deque):

    @staticmethod
    def _get_factor_without_check(db_table, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None,
                                  conds: str = '1', as_alias: (list, tuple, str) = None):
        """

        :param db_table:
        :param factor_names:
        :param cik_dt:
        :param cik_iid:
        :param conds:  conds = @test1>1 | @test2<1
        :return:
        """
        factor_names = FactorCheckHelper.check_factor_names(factor_names)
        alias = FactorCheckHelper.check_alias(factor_names, as_alias=as_alias)
        # rename variables
        f_names_list = [f if (a is None) or (f == a) else f"{f} as {a}" for f, a in zip(factor_names, alias)]
        cols_str = ','.join(f_names_list)

        conditions = '1' if conds == '1' else conds.replace('&', 'and').replace('|', 'or').replace('@', '')
        cik_dt_str = f"{cik_dt} as cik_dt" if cik_dt != 'cik_dt' else cik_dt
        cik_iid_str = f"{cik_iid} as cik_iid" if cik_iid != 'cik_iid' else cik_iid

        sql = f'select {cols_str}, {cik_dt_str}, {cik_iid_str}  from {db_table} where {conditions}'

        return FactorInfo(db_table, cik_dt, cik_iid, ','.join(map(str, factor_names)), ','.join(map(str, alias)),
                          sql, conds)  #

    def show_factors(self, reduced=False, to_df=True):
        if reduced:
            # ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions')
            # ['db_table', 'dts', 'iid', 'conditions']
            cols = list(FactorInfo._fields[:3]) + [FactorInfo._fields[-1]]

            f = pd.DataFrame(self, columns=FactorInfo._fields)
            factor_name_col = FactorInfo._fields[3]
            alias_col = FactorInfo._fields[4]

            # can_merged_index = (fgroupby['sql'].count() > 1).reset_index()
            # can_merged_index = can_merged_index[can_merged_index['sql']]
            # can_merged_index = fgroupby.count().index
            factors = []
            for (db_table, dts, iid, conditions), df in f.groupby(cols):
                # masks = (f['db_table'] == db_table) & (f['dts'] == dts) & (f['iid'] == iid) & (
                #         f['conditions'] == conditions)
                cc = df[[factor_name_col, alias_col]].apply(lambda x: ','.join(x))
                origin_factor_names = cc[factor_name_col].split(',')
                alias = cc[alias_col].split(',')
                origin_factor_names_new, alias_new = zip(*list(set(zip(origin_factor_names, alias))))
                alias_new = list(map(lambda x: x if x != 'None' else None, alias_new))

                # cik_dt, cik_iid = self.check_cik_dt(cik_dt=dts, default_cik_dt=self._cik.dts), self.check_cik_iid(
                #     cik_iid=iid, default_cik_iid=self._cik.iid)
                # add_factor process have checked
                res = self._get_factor_without_check(db_table, origin_factor_names_new, cik_dt=dts, cik_iid=iid,
                                                     conds=conditions, as_alias=alias_new)
                factors.append(res)

        else:
            factors = self
        if to_df:
            return pd.DataFrame(factors, columns=FactorInfo._fields)
        else:
            return factors

    def pull_iter(self, query, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False):
        if not isinstance(query, Callable):
            raise ValueError('query must database connector with __call__')

        factors = self.show_factors(reduced=reduced, to_df=False)

        for db_table, dts, iid, origin_factor_names, alias, sql, conditions in factors:
            ## todo 可能存在性能点
            if add_limit:
                sql2 = f"select * from ({sql}) where {filter_cond_dts} and {filter_cond__ids} limit 100"
            else:
                sql2 = f"select * from ({sql}) where {filter_cond_dts} and {filter_cond__ids}"

            df = query(sql2)
            res = SmartDataFrame(df, db_table, dts, iid, origin_factor_names, alias, sql, conditions).set_index(
                ['cik_dt', 'cik_iid'])

            # ['db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions']
            yield res

    def pull_sql(self, query, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False):
        if not isinstance(query, Callable):
            raise ValueError('query must database connector with __call__')
        factors = self.show_factors(reduced=reduced, to_df=False)
        sql_list = []
        for db_table, dts, iid, origin_factor_names, alias, sql, conditions in factors:
            ## todo 可能存在性能点
            if add_limit:
                sql2 = f"select * from ({sql}) where {filter_cond_dts} and {filter_cond__ids} limit 100"
            else:
                sql2 = f"select * from ({sql}) where {filter_cond_dts} and {filter_cond__ids}"
            sql_list.append(sql2)

        from functools import reduce

        def join(sql1, sql2):
            settings = ' settings join'
            sql = f"select * from ({sql1}) all full join ({sql2}) using (cik_dt,cik_iid)  {settings}"
            return sql

        s = reduce(lambda x, y: join(x, y), sql_list)
        df = query(s)
        res = SmartDataFrame(df, db_table, dts, iid, origin_factor_names, alias, sql, conditions).set_index(
            ['cik_dt', 'cik_iid'])

        # ['db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions']
        yield res


# class BaseSingleFactorTableNode(BaseSingleFactorBaseNode):
#
#     def __init__(self, src: str, db_table: (None, str) = None, info=None,
#                  execute: bool = False,
#                  no_self_update: bool = True, **kwargs):
#         super(BaseSingleFactorTableNode, self).__init__(src, db_table=db_table, info=info, **kwargs)
#         self._complex = False
#         # self._fid_ck = fid_ck
#         # self._dt_max_1st = dt_max_1st
#         self._execute = execute
#         self._no_self_update = no_self_update


class BaseSingleFactorTableNode(FactorCheckHelper, BaseSingleQueryBaseNode):
    __Name__ = "基础因子库单因子表"

    def __init__(self, *args, **kwargs):
        # super(FatctorTable, self).__init__(*args, **kwargs)
        # self._node = BaseSingleFactorBaseNode(*args, **kwargs)
        super(BaseSingleFactorTableNode, self).__init__(*args, **kwargs)

        cik_dt_col = None if 'cik_dt' not in kwargs.keys() else kwargs['cik_dt']
        cik_iid_col = None if 'cik_iid' not in kwargs.keys() else kwargs['cik_iid']
        self._cik_cols = CIK(cik_dt_col, cik_iid_col)

        # self._cik_data = None
        self._checked = False
        self.__auto_check_cik__()
        self._factors = _Factors()

        self._strict_cik = True if 'strict_cik' not in kwargs.keys() else kwargs['strict_cik']
        # self.append = self.add_factor

        self._cik_dts = None
        self._cik_iids = None

    def set_strict_cik(self, strict_cik: bool):
        self._strict_cik = strict_cik

    def __auto_check_cik__(self):
        if not self._checked and (self._cik_cols.dts is None or self._cik_cols.iid is None):
            raise NotImplementedError('cik(dts or iid) is not setup!')
        else:
            self._checked = True

    # def __check_cik__(self, cik_dt=None, cik_iid=None):
    #     if cik_dt is not None:
    #         pass
    #
    #     elif cik_dt is None:
    #         cik_dt = self._cik.dts
    #     else:
    #         raise NotImplementedError('cik_dt is not setup!')
    #
    #     if cik_iid is not None:
    #         pass
    #     elif cik_iid is None :
    #         cik_iid = self._cik.iid
    #     else:
    #         raise NotImplementedError('cik_iid is not setup!')
    #     return cik_dt, cik_iid

    def setup_cik(self, cik_dt_col: str, cik_iid_col: str):
        """
        设置 cik 列名
        :param cik_dt_col:
        :param cik_iid_col:
        :return:
        """

        self._cik_cols = CIK(cik_dt_col, cik_iid_col)

    # def getDB(self, db):
    #     self.db = db

    def show_tables(self, db) -> list:
        """
        show tables from given db
        :param db:
        :return:
        """
        return self.operator(f'show tables from {db}')['name'].unique().tolist()

    def _filter_elements(self, ft_list, rm_cik: bool = True, customized_cik: (None, tuple, list) = None) -> list:
        if rm_cik:
            if customized_cik is None:
                return list(filter(lambda x: x not in self._cik_cols, ft_list))
            elif isinstance(customized_cik, (tuple, list)) and len(customized_cik) == 2:
                return list(filter(lambda x: x not in customized_cik, ft_list))
            else:
                warnings.warn('receive non-standard cik! will use default cik instead!')
                return list(filter(lambda x: x not in self._cik_cols, ft_list))
        else:
            return res

    def show_elements(self, db, table, rm_cik: bool = True, customized_cik: (None, tuple, list) = None) -> list:
        """
        show all elements from given db.table
        :param customized_cik: use custome cik
        :param db:
        :param table:
        :param rm_cik:
        :return:
        """

        res = self.operator(f'desc {db}.{table}')['name'].unique().tolist()
        return self._filter_elements(res, rm_cik=rm_cik, customized_cik=customized_cik)

    def add_factor(self, db_table, factor_names: (list, tuple, str), *, cik_dt=None, cik_iid=None,
                   as_alias: (list, tuple, str) = None):
        conds = '1'  # not allow to set conds
        cik_dt, cik_iid = self.check_cik_dt(cik_dt=cik_dt, default_cik_dt=self._cik_cols.dts), self.check_cik_iid(
            cik_iid=cik_iid, default_cik_iid=self._cik_cols.iid)
        res = self._factors._get_factor_without_check(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
                                                      conds=conds, as_alias=as_alias)
        self._factors.append(res)

    def add_factors_smart(self, *args, **kwargs):

        if len(args) == 2:
            self.add_factor(*args, **kwargs)
        else:
            if len(args) == 1:
                if 'factor_names' in kwargs.keys():
                    self.add_factor(*args, **kwargs)
                else:
                    self.add_factors(*args, **kwargs)
            elif len(args) == 0:
                if 'factor_names' in kwargs.keys():
                    self.add_factor(*args, **kwargs)
                else:
                    self.add_factors(*args, **kwargs)
            else:
                raise ValueError('args got 3 or more values!!')

    def add_factors(self, db: str, cik_dt: (None, str) = None, cik_iid: (None, str) = None,
                    exclude: (None, list) = None, system=True):
        if system:
            self._add_factors_ftmod_system(db, cik_dt=cik_dt, cik_iid=cik_iid,
                                           exclude=exclude, )
        else:
            warnings.warn('will slow down the performance!!!!')
            self._add_factors_ftmod_map(db, cik_dt=cik_dt, cik_iid=cik_iid,
                                        exclude=exclude, )

    def _add_factors_ftmod_system(self, db: str, cik_dt: (None, str) = None, cik_iid: (None, str) = None,
                                  exclude: (None, list) = None):
        exclude = [] if exclude is None else exclude
        if cik_dt is None:
            cik_dt = self._cik_cols.dts
        if cik_iid is None:
            cik_iid = self._cik_cols.iid
        _cik = CIK(cik_dt, cik_iid)

        system_scan_sql = f"select database,table,groupArray(name) as cols from system.columns where database ='{db}' group by database,table "

        for (database, table), cols_df in self.operator(system_scan_sql).groupby(['database', 'table']):
            cols = cols_df['cols'].values.ravel()
            if len(cols) == 1 and isinstance(cols[0], list):
                cols = cols[0]
            else:
                raise ValueError('ClickHouse system.columns sql parsed wrong!')
            factor_prefix = table
            filtered_sub_factors = filter(lambda x: x not in exclude,
                                          self._filter_elements(cols, rm_cik=True,
                                                                customized_cik=_cik))
            sub_factors = list(filtered_sub_factors)
            self.add_factor(db + '_' + table, factor_names=sub_factors, cik_dt=_cik.dts, cik_iid=_cik.iid,
                            as_alias=list(map(lambda x: factor_prefix + '_' + x, sub_factors)))

    def _add_factors_ftmod_map(self, db: str, cik_dt: (None, str) = None, cik_iid: (None, str) = None,
                               exclude: (None, list) = None):
        """


        :param exclude:
        :param db:
        :param cik_dt:
        :param cik_iid:
        :param as_alias:
        :return:
        """
        exclude = [] if exclude is None else exclude
        if cik_dt is None:
            cik_dt = self._cik_cols.dts
        if cik_iid is None:
            cik_iid = self._cik_cols.iid
        _cik = CIK(cik_dt, cik_iid)
        tables = self.show_tables(db)
        from ClickSQL.utils.process_bar import process_bar
        for table in process_bar(tables):
            factor_prefix = table
            filtered_sub_factors = filter(lambda x: x not in exclude,
                                          self.show_elements(db, table, rm_cik=True,
                                                             customized_cik=_cik))
            sub_factors = list(filtered_sub_factors)
            self.add_factor(db + '_' + table, factor_names=sub_factors, cik_dt=_cik.dts, cik_iid=_cik.iid,
                            as_alias=list(map(lambda x: factor_prefix + '_' + x, sub_factors)))
            time.sleep(0.1)

        # conds = '1'  # not allow to set conds
        # cik_dt, cik_iid = self.check_cik_dt(cik_dt=cik_dt, default_cik_dt=self._cik_cols.dts), self.check_cik_iid(
        #     cik_iid=cik_iid, default_cik_iid=self._cik_cols.iid)
        # all_cols = self._node('desc {db_table}')
        #
        # factor_names = list(filter(lambda x: x not in (cik_dt, cik_iid), all_cols['name'].unique().tolist()))
        # res = self._factors._get_factor_without_check(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
        #                                               conds=conds, as_alias=as_alias)
        # self._factors.append(res)

    def show_factors(self, reduced=False, to_df=True):
        return self._factors.show_factors(reduced=reduced, to_df=to_df)

        # no_duplicates_df = f.eval("+".join(cols))
        ## todo auto merge same condition,dbtable,dts,iid
        # return can_merged_index

    def __iter__(self):
        return self._factors.pull_iter(self._node, self._cik_dt_str, self._cik_iid_str, reduced=True,
                                       add_limit=False)

    def __str__(self):
        return '\n'.join(self.show_factors(reduced=True)['sql'].unique().tolist())

    def head(self, reduced=True, ):
        """
        quick look top data
        :param reduced:
        :return:
        """

        return self.pull(reduced=reduced, add_limit=True)

    def pull(self, reduced=True, add_limit=False, strict_cik=True):
        if self._strict_cik or strict_cik:
            if self._cik_dts is None:
                raise KeyError('datetime data is not setup!')
            if self._cik_iids is None:
                raise KeyError('id data is not setup!')

        return pd.concat(
            self._factors.pull_iter(self._node, self._cik_dt_str, self._cik_iid_str, reduced=reduced,
                                    add_limit=add_limit), axis=1)

    @property
    def cik(self):
        return CIK(self._cik_dts, self._cik_iids)

    @property
    def _cik_dt_str(self):
        """
        set cik_dt

        :return:
        """
        dt_format = "%Y%m%d"
        if self._cik_dts is None:
            return "  1 "

        else:
            cik_dts_str = "','".join(map(lambda x: x.strftime(dt_format), pd.to_datetime(self._cik_dts)))
            return f" cik_dt in ('{cik_dts_str}') "

    # @_cik_dt_str.setter
    def set_cik_dt(self, cik_dt: list):
        self._cik_dts = cik_dt

    @property
    def _cik_iid_str(self):
        if self._cik_iids is None:
            return "  1 "

        else:
            cik_iid_str = "','".join(map(lambda x: x, self._cik_iids))
            return f" cik_iid in ('{cik_iid_str}') "

    # @_cik_iid_str.setter
    def set_cik_iid(self, cik_iid: list):
        self._cik_iids = cik_iid


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
