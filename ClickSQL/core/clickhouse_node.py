# coding=utf-8
"""
Created on Sun Aug 18 12:17:40 2019

@author: lee1984 & snowfree
"""
import copy
import gzip
import json
import re
import urllib
import warnings
from collections import ChainMap, namedtuple
from functools import lru_cache

import grequests
import numpy as np
import pandas as pd
import requests

from Nodes.conf_node.parse_rfc_1738_args import _parse_rfc1738_args
from Nodes.utils_node.lazy_load import LazyInit

node = namedtuple('clickhouse', ['host', 'port', 'user', 'password', 'database'])


class DBUtilsCheck(object):
    @staticmethod
    def _generate_query_tools_(obj, sql: str, return_full_res: bool = False):
        res = obj.query(sql)
        if res.empty:
            return False
        else:
            if return_full_res:
                return res
            else:
                return True

    @classmethod
    def _check_database_exists(cls, obj: object, db: str, return_full_res: bool = True):
        """
        :param db:
        :param table:
        :param obj:
        :return: exists =true not exists = False
        """
        sql = f"select * from system.databases where name = '{db}'  "
        return cls._generate_query_tools_(obj, sql, return_full_res=return_full_res)

    @classmethod
    def _check_table_exists(cls, obj: object, db: str, table: str, return_full_res: bool = True):
        """
        :param db:
        :param table:
        :param obj:
        :return: exists =true not exists = False
        """
        sql = f"select * from system.tables where database = '{db}' and name = '{table}'  "
        return cls._generate_query_tools_(obj, sql, return_full_res=return_full_res)

    @classmethod
    def _check_table_empty(cls, obj: object, db: str, table: str, return_full_res: bool = True):
        sql = f"select count(*) from {db}.{table}"
        return cls._generate_query_tools_(obj, sql, return_full_res=return_full_res)

    # @staticmethod
    # def check_db_settings(settings: str):
    #     if isinstance(settings, str):
    #         db_type, settings = ConnectionParser.parser(settings)
    #         if db_type.lower() != 'clickhouse':
    #             raise ValueError('settings is not for clickhouse!')
    #     elif isinstance(settings, dict):
    #         db_type = 'Clickhouse'
    #     else:
    #         raise ValueError('settings must be str or dict')
    #     return db_type, settings


class DBUtilsGet(DBUtilsCheck):
    @staticmethod
    def _get_table_engine(obj: object, db: str, table: str):
        sql = f"select engine from system.tables where database = '{db}' and name = {table} "
        res = obj.query(sql)
        if res.empty:
            raise ValueError(f'Cannot locate table: {db}.{table}!')
        else:
            return res['engine'].values


class DBUtilsCreate(DBUtilsGet):
    @staticmethod
    def detect_end_with_limit(string, pattern=r'[\s]+limit[\s]+[0-9]+$'):
        m = re.findall(pattern, string)
        if m is None or m == []:
            return False
        else:
            return True

    @staticmethod
    def translate_dtypes_from_df(df: pd.DataFrame,
                                 translate_dtypes: dict = {'object': 'String',
                                                           'datetime64[ns]': 'Datetime'}):
        if hasattr(df, 'dtypes'):
            dtypes_series = df.dtypes.replace(translate_dtypes)
            return dtypes_series.map(lambda x: str(x).capitalize()).to_dict()
        elif hasattr(df, '_columns_') and 'type' in df._columns_ and 'name' in df._columns_:
            dtypes_series = df.set_index('name')['type'].replace(translate_dtypes)
            return dtypes_series.map(lambda x: str(x)).to_dict()
        else:
            raise ValueError(f'unknown df:{type(df)}')

    @staticmethod
    def translate_dtype1_as_dtype2(df: pd.DataFrame, src2target={'category': 'str'}):
        dtypes_series = df.dtypes
        for src, dest in src2target.items():
            if src in dtypes_series:
                category_cols = dtypes_series[dtypes_series == src].index
                for col in category_cols:
                    df[col] = df[col].astype(dest)
            else:
                pass
        return df

    @classmethod
    def _create_table_from_df(cls, obj: object, db: str, table: str, df: pd.DataFrame, key_cols: (list, tuple),
                              engine_type: str = 'ReplacingMergeTree', extra_format_dict=None, partitions_expr=None):
        query_func = obj.query

        df = cls.translate_dtype1_as_dtype2(df, src2target={'category': 'str'})
        cols = df.columns
        dtypes_dict = cls.translate_dtypes_from_df(df)
        if extra_format_dict is None:
            pass
        else:
            dtypes_dict.update(extra_format_dict)
        dtypes_dict = {k: v for k, v in dtypes_dict.items() if k in cols}
        base = cls._create_table_from_sql(db, table, dtypes_dict, key_cols, engine_type=engine_type,
                                          extra_format_dict=extra_format_dict, partitions_expr=partitions_expr)
        exist_status = cls._check_table_exists(obj, db, table)

        query_func(base)
        return exist_status

    @classmethod
    def _create_table_sql(cls, db: str, table: str, dtypes_dict: dict, key_cols: (list, tuple),
                          engine_type: str = 'ReplacingMergeTree', partitions_expr=None):
        # dtypes_dict.update(extra_format_dict)
        cols_def = ','.join([f"{name} {d_type}" for name, d_type in dtypes_dict.items()])
        order_by_cols = ','.join(key_cols)

        maid_body = f"CREATE TABLE IF NOT EXISTS {db}.{table} ( {cols_def} ) ENGINE = {engine_type}"
        settings = "SETTINGS index_granularity = 8192"
        conds = f"ORDER BY ( {order_by_cols} )"
        if partitions_expr is None:
            partitions = ''
        else:
            partitions = f"PARTITION BY {partitions_expr}"
        base = f"{maid_body} {conds} {partitions} {settings}"
        return base

    @classmethod
    def _create_table_from_sql(cls, obj: object, db: str, table: str, sql: str, key_cols: list,
                               engine_type: str = 'ReplacingMergeTree',
                               extra_format_dict: (dict, None) = None,
                               partitions_expr=None) -> bool:
        """

        :param obj:
        :param db:
        :param table:
        :param sql:
        :param key_cols:
        :param engine_type:
        :param extra_format_dict:
        :return:
        """

        if extra_format_dict is None:
            extra_format_dict = {}

        query_func = obj.query
        if sql.endswith(';'):
            sql = sql[:-1]
        end_with_limit_status = cls.detect_end_with_limit(sql, pattern=r'[\s]+limit[\s]+[0-9]+$')
        if end_with_limit_status:
            describe_sql = f' describe({sql}) '
        else:
            describe_sql = f'describe ( {sql} limit 1)'

        exist_status = cls._check_table_exists(obj, db, table)
        if exist_status:
            print('table:{table} already exists!')
        else:
            print('will create {table} at {db}')
            dtypes_df = query_func(describe_sql)
            dtypes_dict = dict(dtypes_df[['name', 'type']].drop_duplicates().values)
            if extra_format_dict is None:
                pass
            else:
                dtypes_dict.update(extra_format_dict)
            sql = cls._create_table_sql(db, table, dtypes_dict, key_cols, engine_type=engine_type,
                                        partitions_expr=partitions_expr)
            query_func(sql)
        return exist_status

    @classmethod
    def _create_table(cls, obj: object, db: str, table: str, sql: str, key_cols: list,
                      engine_type: str = 'ReplacingMergeTree',
                      extra_format_dict: (dict, None) = None, partitions_expr: (str, None) = None) -> object:
        if isinstance(sql, str):
            return cls._create_table_from_sql(obj, db, table, sql, key_cols,
                                              engine_type=engine_type,
                                              extra_format_dict=extra_format_dict, partitions_expr=partitions_expr)
        elif isinstance(sql, pd.DataFrame):
            return cls._create_table_from_df(obj, db, table, sql, key_cols,
                                             engine_type='ReplacingMergeTree',
                                             extra_format_dict=extra_format_dict, partitions_expr=partitions_expr)
        else:
            raise ValueError(f'unknown sql:{sql}')


class DBUtilsCore(DBUtilsCreate):
    @staticmethod
    def _check_df_and_dump(df, describe_table):
        non_nullable_columns = list(describe_table[~describe_table['type'].str.startswith('Nullable')]['name'])
        integer_columns = list(describe_table[describe_table['type'].str.contains('Int', regex=False)]['name'])
        missing_in_df = {i: np.where(df[i].isnull(), 1, 0).sum() for i in non_nullable_columns}

        df_columns = list(df.columns)
        each_row = df.to_dict(orient='records')
        for i in missing_in_df:
            if missing_in_df[i] > 0:
                raise ValueError('"{0}" is not a nullable column, missing values are not allowed.'.format(i))

        for row in each_row:
            for col in df_columns:
                if pd.isnull(row[col]):
                    row[col] = None
                else:
                    if col in integer_columns:
                        try:
                            row[col] = int(row[col])
                        except Exception as e:
                            print(str(e))
                            raise ValueError('Column "{0}" is {1}, while value "{2}"'.format(col,
                                                                                             describe_table[
                                                                                                 describe_table[
                                                                                                     'name'] == col].iloc[
                                                                                                 0]['type'], row[col]) + \
                                             ' in the dataframe column cannot be converted to Integer.')
            yield json.dumps(row, ensure_ascii=False)

    @classmethod
    def _merge_settings(cls, settings, updated_settings=None):
        """

        :param settings:
        :param updated_settings:
        :return:
        """
        if updated_settings is None:
            updated_settings = {'enable_http_compression': 1, 'send_progress_in_http_headers': 0,
                                'log_queries': 1, 'connect_timeout': 10, 'receive_timeout': 300,
                                'send_timeout': 300, 'output_format_json_quote_64bit_integers': 0,
                                'wait_end_of_query': 0}

        if settings is not None:
            invalid_setting_keys = list(set(settings.keys()) - set(updated_settings.keys()))
            if len(invalid_setting_keys) > 0:
                raise ValueError('setting "{0}" is invalid, valid settings are: {1}'.format(
                    invalid_setting_keys[0], ', '.join(updated_settings.keys())))
            else:
                pass
            updated_settings.update(settings)

        return {k: v * 1 if isinstance(v, bool) else v for k, v in updated_settings.items()}

    @staticmethod
    def _check_sql_select_only(sql):
        if sql.strip(' \n\t').lower()[:4] not in ['sele', 'desc', 'show', 'opti', 'crea']:
            raise ValueError('"query" should start with "select" or "describe" or "show", ' + \
                             'while the provided "query" starts with "{0}"'.format(sql.strip(' \n\t').split(' ')[0]))

    @staticmethod
    def _transfer_sql_format(sql, convert_to, transfer_sql_format=True):
        if transfer_sql_format:
            clickhouse_format = 'JSON' if convert_to is None else 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
            query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')
            return query_with_format
        else:
            return sql

    @staticmethod
    def _load_into_pd(ret_value, convert_to: str = 'dataframe'):
        if convert_to.lower() == 'dataframe':
            result_dict = json.loads(ret_value, strict=False)
            meta = result_dict['meta']
            name = map(lambda x: x['name'], meta)
            data = result_dict['data']
            df = pd.DataFrame.from_records(data, columns=list(name))

            for i in meta:
                if i['type'] in ['DateTime', 'Nullable(DateTime)']:
                    df[i['name']] = pd.to_datetime(df[i['name']])
            ret_value = df
        return ret_value


class TableEngineCreator(object):
    @staticmethod
    def _assemble_cols_2_clause(prefix, cols, default=''):
        if cols is None:
            return default
        else:
            cols_str = ','.join(cols)
            return f"{prefix} ( {cols_str} ) "

    @classmethod
    def ReplacingMergeTree_creator(cls, DB_TABLE, cols_def, order_by_cols,
                                   sample_by_cols=None,
                                   ON_CLUSTER='', partition_by_cols=None, primary_by_cols=None):

        order_by_cols_str = ','.join(order_by_cols)
        ORDER_BY_CLAUSE = f'ORDER BY ( {order_by_cols_str} )'

        SAMPLE_CLAUSE = cls._assemble_cols_2_clause('SAMPLE BY', sample_by_cols, default='')

        PRIMARY_BY_CLAUSE = cls._assemble_cols_2_clause('PRIMARY BY', primary_by_cols, default='')
        # if primary_by_cols is not None:
        #     primary_by_cols_str = ','.join(primary_by_cols)
        #     PRIMARY_BY_CLAUSE = f'PRIMARY BY ( {primary_by_cols_str} )'
        # else:
        #     PRIMARY_BY_CLAUSE = ''

        PARTITION_by_CLAUSE = cls._assemble_cols_2_clause('PARTITION BY', partition_by_cols, default='')

        # if partition_by_cols is not None:
        #     partition_by_cols_str = ','.join(partition_by_cols)
        #     PARTITION_by_CLAUSE = f'PARTITION BY ( {partition_by_cols_str} )'
        # else:
        #     PARTITION_by_CLAUSE = ''

        return cls.raw_create_ReplacingMergeTree_table_sql(DB_TABLE, cols_def, ORDER_BY_CLAUSE,
                                                           PRIMARY_BY_CLAUSE=PRIMARY_BY_CLAUSE,
                                                           SAMPLE_CLAUSE=SAMPLE_CLAUSE,
                                                           ENGINE_TYPE='ReplacingMergeTree', ON_CLUSTER=ON_CLUSTER,
                                                           PARTITION_by_CLAUSE=PARTITION_by_CLAUSE)

    @staticmethod
    def raw_create_ReplacingMergeTree_table_sql(DB_TABLE, cols_def, ORDER_BY_CLAUSE,
                                                PRIMARY_BY_CLAUSE='', SAMPLE_CLAUSE='',
                                                ENGINE_TYPE='ReplacingMergeTree', ON_CLUSTER='', PARTITION_by_CLAUSE='',
                                                TTL=''
                                                ):
        ## TODO add ttl expr at future
        """

        :param ON_CLUSTER:
        :param SAMPLE_CLAUSE:
        :param PRIMARY_BY_CLAUSE:
        :param PARTITION_by_CLAUSE:
        :param DB_TABLE:
        :param cols_def:
        :param ORDER_BY_CLAUSE:
        :param ENGINE_TYPE:
        :return:
        """
        """CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
            (
                name1 [type1] [DEFAULT|MATERIALIZED|ALIAS expr1],
                name2 [type2] [DEFAULT|MATERIALIZED|ALIAS expr2],
                ...
            ) ENGINE = ReplacingMergeTree([ver])
            [PARTITION BY expr]
            [ORDER BY expr]
            [PRIMARY KEY expr]
            [SAMPLE BY expr]
            [SETTINGS name=value, ...]"""

        maid_body = f"CREATE TABLE IF NOT EXISTS {DB_TABLE} {ON_CLUSTER} ( {cols_def} ) ENGINE = {ENGINE_TYPE}"

        settings = "SETTINGS index_granularity = 8192"
        conds = f"{PARTITION_by_CLAUSE} {ORDER_BY_CLAUSE} {PRIMARY_BY_CLAUSE} {SAMPLE_CLAUSE}"

        base = f"{maid_body} {conds}  {settings}"
        return base


class SQLBuilder(TableEngineCreator):
    @staticmethod
    def _assemble_sample(sample=None):
        if sample is None:
            SAMPLE_CLAUSE = ''
        else:
            SAMPLE_CLAUSE = f'SAMPLE {sample}'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_array_join(array_join_list=None):
        if array_join_list is None:
            ARRAY_JOIN_CLAUSE = ''
        else:
            array_join = ','.join(array_join_list)
            ARRAY_JOIN_CLAUSE = f'ARRAY JOIN {array_join}'
        return ARRAY_JOIN_CLAUSE

    @staticmethod
    def _assemble_join(join_info_dict=None):

        if join_info_dict is None:
            JOIN_CLAUSE = ''
        else:
            join_type = join_info_dict.get('type')
            on_ = join_info_dict.get('ON')
            using_ = join_info_dict.get('USING')

            if join_type is None:
                raise ValueError('join_info_dict cannot locate join_type condition')

            if on_ is None:
                if using_ is None:
                    raise ValueError('join_info_dict cannot locate ON or USING condition')
                else:
                    JOIN_CLAUSE = f'{join_type} USING ({using_})'
            else:
                JOIN_CLAUSE = f'{join_type} ON {on_}'
        return JOIN_CLAUSE

    @staticmethod
    def _assemble_where_like(a_list, prefix='WHERE'):
        if a_list is None:
            SAMPLE_CLAUSE = ''
        else:
            a_list_str = ' and '.join(a_list)
            SAMPLE_CLAUSE = f'{prefix} {a_list_str}'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_group_by(group_by_cols=None):
        if group_by_cols is None:
            SAMPLE_CLAUSE = ''
        else:
            group_by_cols_str = ','.join(group_by_cols)
            SAMPLE_CLAUSE = f'GROUP BY ({group_by_cols_str})'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_order_by(order_by_cols=None):
        if order_by_cols is None:
            SAMPLE_CLAUSE = ''
        else:
            order_by_cols_str = ','.join(order_by_cols)
            SAMPLE_CLAUSE = f'ORDER BY ({order_by_cols_str})'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_limit_by(limit_n_by_dict=None):

        if limit_n_by_dict is None:
            SAMPLE_CLAUSE = ''
        else:
            N = limit_n_by_dict['N']
            order_by_cols_str = ','.join(limit_n_by_dict['limit_by_cols'])
            SAMPLE_CLAUSE = f'LIMIT {N} BY ({order_by_cols_str})'
        return SAMPLE_CLAUSE

    @staticmethod
    def _assemble_limit(limit_n=None):

        if limit_n is None:
            SAMPLE_CLAUSE = ''
        else:
            SAMPLE_CLAUSE = f'LIMIT {limit_n} '
        return SAMPLE_CLAUSE

    @staticmethod
    def raw_create_select_sql(SELECT_CLAUSE: str, DB_TABLE: str, SAMPLE_CLAUSE: str, ARRAY_JOIN_CLAUSE: str,
                              JOIN_CLAUSE: str, PREWHERE_CLAUSE: str, WHERE_CLAUSE: str, GROUP_BY_CLAUSE: str,
                              HAVING_CLAUSE: str, ORDER_BY_CLAUSE: str, LIMIT_N_CLAUSE: str, LIMIT_CLAUSE: str):
        """

        :param SELECT_CLAUSE:
        :param DB_TABLE:
        :param SAMPLE_CLAUSE:
        :param ARRAY_JOIN_CLAUSE:
        :param JOIN_CLAUSE:
        :param PREWHERE_CLAUSE:
        :param WHERE_CLAUSE:
        :param GROUP_BY_CLAUSE:
        :param HAVING_CLAUSE:
        :param ORDER_BY_CLAUSE:
        :param LIMIT_N_CLAUSE:
        :param LIMIT_CLAUSE:
        :return:
        """
        """SELECT [DISTINCT] expr_list
                    [FROM [db.]table | (subquery) | table_function] [FINAL]
                    [SAMPLE sample_coeff]
                    [ARRAY JOIN ...]
                    [GLOBAL] ANY|ALL INNER|LEFT JOIN (subquery)|table USING columns_list
                    [PREWHERE expr]
                    [WHERE expr]
                    [GROUP BY expr_list] [WITH TOTALS]
                    [HAVING expr]
                    [ORDER BY expr_list]
                    [LIMIT n BY columns]
                    [LIMIT [n, ]m]
                    [UNION ALL ...]
                    [INTO OUTFILE filename]
                    [FORMAT format]"""
        if DB_TABLE.lower().startswith('select '):
            DB_TABLE = f"( {DB_TABLE} )"
        else:
            pass
        main_body = f"SELECT {SELECT_CLAUSE} FROM {DB_TABLE} {SAMPLE_CLAUSE}"
        join = f"{ARRAY_JOIN_CLAUSE} {JOIN_CLAUSE}"
        where_conditions = f"{PREWHERE_CLAUSE} {WHERE_CLAUSE} {GROUP_BY_CLAUSE} {HAVING_CLAUSE} "
        order_limit = f"{ORDER_BY_CLAUSE} {LIMIT_N_CLAUSE} {LIMIT_CLAUSE}"
        sql = f"{main_body} {join} {where_conditions} {order_limit}"
        return sql

    @classmethod
    def create_select_sql(cls, DB_TABLE: str, cols: list,
                          sample: (int, float, None) = None,
                          array_join: (list, None) = None, join: (dict, None) = None,
                          prewhere: (list, None) = None, where: (list, None) = None, having: (list, None) = None,
                          group_by: (list, None) = None,
                          order_by: (list, None) = None, limit_by: (dict, None) = None,
                          limit: (int, None) = None) -> str:
        """

        :param having: str ["r1 >1 and r2 <2"]
        :param DB_TABLE: str default.test
        :param cols: list [ r1,r2,r3 ]
        :param sample: str 0.1 or 1000
        :param array_join: list ['arrayA as a','arrayB as b']
        :param join: dict {'type':'all left join','USING' : "r1,r2"}
        :param prewhere: str ["r1 >1" , "r2 <2"]
        :param where: str ["r1 >1.5" , "r2 <1.3"]
        :param group_by: list ['r1','r2']
        :param order_by: list ['r1 desc','r2 desc']
        :param limit_by: dict {'N':10,'limit_by_cols':['r1','r2']}
        :param limit: int 100
        :return:  str
        """

        SELECT_CLAUSE = ','.join(cols)
        SAMPLE_CLAUSE = cls._assemble_sample(sample=sample)
        ARRAY_JOIN_CLAUSE = cls._assemble_array_join(array_join_list=array_join)
        JOIN_CLAUSE = cls._assemble_join(join)
        PREWHERE_CLAUSE = cls._assemble_where_like(prewhere, prefix='PREWHERE')
        WHERE_CLAUSE = cls._assemble_where_like(where, prefix='WHERE')
        HAVING_CLAUSE = cls._assemble_where_like(having, prefix='HAVING')
        GROUP_BY_CLAUSE = cls._assemble_group_by(group_by)
        ORDER_BY_CLAUSE = cls._assemble_order_by(order_by)
        LIMIT_N_CLAUSE = cls._assemble_limit_by(limit_by)
        LIMIT_CLAUSE = cls._assemble_limit(limit)

        return cls.raw_create_select_sql(SELECT_CLAUSE, DB_TABLE, SAMPLE_CLAUSE, ARRAY_JOIN_CLAUSE, JOIN_CLAUSE,
                                         PREWHERE_CLAUSE, WHERE_CLAUSE, GROUP_BY_CLAUSE, HAVING_CLAUSE, ORDER_BY_CLAUSE,
                                         LIMIT_N_CLAUSE, LIMIT_CLAUSE)

    # @classmethod
    # def group_by(cls, base_sql: str, by: list, agg_cols: list, where: list = None, having: list = None, order_by=None,
    #              limit_by=None, limit=None):
    #     sql = cls.create_select_sql(DB_TABLE=base_sql, cols=by + agg_cols,
    #                                 sample=None, array_join=None, join=None,
    #                                 prewhere=None, where=where, having=having,
    #                                 group_by=by, order_by=order_by, limit_by=limit_by, limit=limit)
    #
    #     return sql


class ClickHouseBaseNodeQuery(DBUtilsCore, SQLBuilder):
    accepted_formats = ['DataFrame', 'TabSeparated', 'TabSeparatedRaw', 'TabSeparatedWithNames',
                        'TabSeparatedWithNamesAndTypes', 'CSV', 'CSVWithNames', 'Values', 'Vertical', 'JSON',
                        'JSONCompact', 'JSONEachRow', 'TSKV', 'Pretty', 'PrettyCompact',
                        'PrettyCompactMonoBlock', 'PrettyNoEscapes', 'PrettySpace', 'XML']
    _default_settings = {'enable_http_compression': 1, 'send_progress_in_http_headers': 0,
                         'log_queries': 1, 'connect_timeout': 10, 'receive_timeout': 300,
                         'send_timeout': 300, 'output_format_json_quote_64bit_integers': 0,
                         'wait_end_of_query': 0}

    def __init__(self, **db_settings):
        """
        :param db_settings:
        """

        self._db = db_settings['database']

        self._para = node(db_settings['host'], db_settings['port'], db_settings['username'],
                          db_settings['password'], db_settings['database'])

        self._base_url = "http://{host}:{port}/?".format(host=self._para.host, port=int(self._para.port))

        self.http_settings = self._merge_settings(None, updated_settings=self._default_settings)
        self.http_settings.update({'user': self._para.user, 'password': self._para.password})
        self._async = False
        self._session = None
        self.max_async_query_once = 5
        self.is_closed = False
        self._test_connection_()

    def _test_connection_(self):
        ret_value = self.session.get(self._base_url)

        print('test_connection: ', ret_value.text.strip())

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        else:
            if self.is_closed:
                raise ValueError('session is closed!')
            else:
                pass
        return self._session  # grequests.AsyncRequest

    def __setitem__(self, db_table: str, df):
        if '.' in db_table:
            pass
            db, table = db_table.split('.')
        else:
            raise ValueError(f'get unknown db_table : {db_table}')
        self._df_insert_(db, table, df)

    def close(self):
        self.session.close()
        self.is_closed = True  # self._session.is_closed()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __desc__(self, db: str, table: str):
        describe_sql = f'describe table {db}.{table}'
        describe_table = self._request(describe_sql)
        # non_nullable_columns = list(describe_table[~describe_table['type'].str.startswith('Nullable')]['name'])
        # integer_columns = list(describe_table[describe_table['type'].str.contains('Int', regex=False)]['name'])
        # missing_in_df = {i: np.where(df[i].isnull(), 1, 0).sum() for i in non_nullable_columns}
        #
        # df_columns = list(df.columns)
        # each_row = df.to_dict(orient='records')
        # del df
        return describe_table  # , integer_columns, non_nullable_columns

    def _request_unit_(self, sql: str, convert_to: str = 'dataframe', transfer_sql_format: bool = True):
        sql2 = self._transfer_sql_format(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format)

        if self._async:
            session = grequests
        else:
            session = self.session

        if self.http_settings['enable_http_compression'] == 1:
            url = self._base_url + urllib.parse.urlencode(self.http_settings)
            resp = session.post(url,
                                data=gzip.compress(sql2.encode()),
                                headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        else:
            # settings = self.http_settings.copy()
            # settings.update({'query': sql2})
            url = self._base_url + urllib.parse.urlencode(ChainMap(self.http_settings, {'query': sql2}))
            resp = session.post(url)
        return resp

    # @timer
    # @lru_cache(maxsize=10)
    def _request(self, sql: str, convert_to: str = 'dataframe', todf: bool = True, transfer_sql_format: bool = True):
        if self._async:
            self._async = False
        resp = self._request_unit_(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format)
        if todf:
            d = self._load_into_pd(resp.content, convert_to)
            return d
        else:
            return resp.content

    def _async_request(self, sql_list, convert_to='dataframe', todf=True, transfer_sql_format=True, auto_switch=True):

        def exception_handler(request, exception):
            print("Request failed")

        if not self._async:
            if auto_switch:
                self._async = True
            else:
                raise ValueError("must manually switch to async mode")
        tasks = (self._request_unit_(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format) for sql in
                 sql_list)

        for resp in grequests.map(tasks, exception_handler=exception_handler, size=self.max_async_query_once):
            if todf:
                d = self._load_into_pd(resp.content, convert_to)
                yield d
            else:
                yield resp.content
        if auto_switch:
            self._async = False

    # @timer
    def _async_query(self, sql_list: (list,)):

        get_query = all(map(
            lambda sql: sql.lower().startswith('select') or sql.lower().startswith('show') or sql.lower().startswith(
                'desc'), sql_list))
        if get_query:
            todf = True
        else:
            insert_query = all(map(lambda sql: sql.lower().startswith('insert') or sql.lower().startswith(
                'optimize') or sql.lower().startswith('create'), sql_list))
            # res = self._request(sql, convert_to='dataframe', todf=True)
            if insert_query:
                todf = False
                # res = self._request(sql, convert_to='dataframe', todf=False)
            else:
                raise ValueError('Unknown sql! current only accept select, insert, show, optimize')
        res = list(self._async_request(sql_list, convert_to='dataframe', todf=todf))
        return res

    def _df_insert_(self, db: str, table: str, df: pd.DataFrame):
        describe_table = self.__describe__(db, table)
        query_with_format = 'insert into {0} format JSONEachRow \n{1}'.format('{}.{}'.format(db, table), '\n'.join(
            self._check_df_and_dump(df, describe_table)))
        self._request(query_with_format, convert_to='dataframe', todf=False, transfer_sql_format=False)
    @lru_cache(maxsize=100)
    def execute(self, sql: str, available_queries_select=['select', 'show', 'desc'],
                available_queries_insert=['insert', 'optimize', 'create']):
        if isinstance(sql, str):
            return self._execute_(sql,
                                  available_queries_select=available_queries_select,
                                  available_queries_insert=available_queries_insert)
        elif isinstance(sql, (list, tuple)):
            max_queries = self.max_async_query_once * 2
            if len(sql) > max_queries:
                raise ValueError(f'too many queries,please reduce to less than {max_queries}!')
            return self._async_query(sql)
        else:
            raise ValueError('sql must be str or list or tuple')

    def _execute_(self, sql: str,
                  available_queries_select=['select', 'show', 'desc'],
                  available_queries_insert=['insert', 'optimize', 'create']):

        head = sql.lower()[:4]

        if head in map(lambda x: x[:4], available_queries_select):
            todf = True
            transfer_sql_format = True
        elif head in map(lambda x: x[:4], available_queries_insert):
            todf = False
            transfer_sql_format = False
        else:
            raise ValueError('Unknown sql! current only accept {}'.format(
                ','.join(available_queries_select + available_queries_insert)))
        res = self._request(sql, convert_to='dataframe', todf=todf, transfer_sql_format=transfer_sql_format)
        return res
    @lru_cache(maxsize=100)
    def query(self, sql: str, optimize: bool = False):
        """

        :param sql:
        :param optimize:
        :return:
        """
        if isinstance(sql, str) and sql.lower().startswith('insert into'):
            db_table = sql.lower().split('insert into ')[-1].split(' ')[0]
        else:
            db_table = 'no'
        res = self.execute(sql)
        try:
            if optimize and db_table != 'no':
                self.execute(f'optimize table {db_table}')
        except Exception as e:
            warnings.warn(f'auto optimize process failure, please manual optimize on {db_table}')
        finally:
            return res


class ClickHouseTableNode(LazyInit, ClickHouseBaseNodeQuery):
    def __init__(self, table_name, **settings):
        super(ClickHouseTableNode, self).__init__(**settings)
        self.table_name = table_name

    @property
    def tables(self):
        sql = 'SHOW TABLES FROM {db}'.format(db=self._db)
        res = self.query(sql).values.ravel().tolist()
        return res

    @property
    def databases(self):
        sql = 'SHOW DATABASES'
        res = self.query(sql).values.ravel().tolist()
        return res

    def create_table(self, db: str, table: str, select_sql: str, keys_cols: list,
                     table_engine: str = 'ReplacingMergeTree', extra_format_dict: bool = None,
                     return_status: bool = True, partitions_expr=None):
        status = self._create_table(self, db, table, select_sql, keys_cols, engine_type=table_engine,
                                    extra_format_dict=extra_format_dict,
                                    partitions_expr=partitions_expr)
        if return_status:
            return status

    @property
    def _table_exist_status(self):
        return self.table_name in self.tables

    @property
    def table_structure(self):
        if self._table_exist_status:
            return self.__desc__(self._db, self.table_name)
        else:
            raise ValueError(f'table: {self.table_name} may not be exists!')

    @property
    def columns(self):
        return self.table_structure['name'].values.ravel().tolist()

    def head(self, top=10):
        sql = f'select * from {self._db}.{self.table_name} limit {top}'
        return self.query(sql, optimize=False)


class ClickHouseTableNodeSQLNode(object):
    def __init__(self, table_name, settings):
        if isinstance(settings, dict):
            pass
        elif isinstance(settings, str):
            settings = _parse_rfc1738_args(settings)
        else:
            raise ValueError(f'unknown settings: {settings}')

        self._node = ClickHouseTableNode(table_name, **settings)
        self.db = self._node._db
        self.table = table_name
        self.tables = self._node.tables
        self.databases = self._node.databases
        self.query=self._node.query

        self._is_sql = False
        self._temp_sql = None
        self._columns = self._node.columns
        self._sample = None
        self._join = None
        self._array_join = None
        self._where = None
        self._pre_where = None
        self._group_by = None
        self._order_by = None
        self._limit_by = None
        self._limit = None
        self._having = None

    def groupby(self, by: list, agg_cols: list, where: list = None, having: list = None, order_by=None,
                limit_by=None, limit=None):
        a = copy.deepcopy(self)
        a._columns_ = by + agg_cols
        a._where_ = where
        a._having_ = having
        a._group_by_ = by
        a._order_by_ = order_by
        a._limit_by_ = limit_by
        a._limit_ = limit
        return a

    @property
    def _base_(self):
        if self._is_sql and self._temp_sql is not None:
            return "( " + self._temp_sql + " )"
        elif self._is_sql and self._temp_sql is None:
            raise ValueError('sql format has been translated, but got None sql')
        else:
            return self.db_table

    @property
    def _sql_(self):
        DB_TABLE = self._base_
        cols = self._columns_
        sample = self._sample_
        array_join = self._array_join_
        join = self._join_
        pre_where = self._pre_where_
        where = self._where_
        having = self._having_
        group_by = self._group_by_
        order_by = self._order_by_
        limit_by = self._limit_by_
        limit = self._limit_
        sql = self.create_select_sql(DB_TABLE, cols,
                                     sample=sample,
                                     array_join=array_join, join=join,
                                     prewhere=pre_where, where=where, having=having,
                                     group_by=group_by,
                                     order_by=order_by, limit_by=limit_by,
                                     limit=limit)
        return sql.strip()

    def __query__(self):
        sql = self._sql_

        return self.query(sql + ' limit 10')

    @property
    def db_table(self):
        return "{}.{}".format(self.db, self.table)

    @property
    def _columns_(self):
        return self._columns

    @_columns_.setter
    def _columns_(self, cols):
        self._columns = cols

    @property
    def _sample_(self):
        return self._sample

    @_sample_.setter
    def _sample_(self, samples):
        self._sample = samples

    @property
    def _join_(self):
        return self._join

    @_join_.setter
    def _join_(self, joins):
        self._join = joins

    @property
    def _array_join_(self):
        return self._array_join

    @_array_join_.setter
    def _array_join_(self, joins):
        self._array_join = joins

    @property
    def _where_(self):
        return self._where

    @_where_.setter
    def _where_(self, wheres):
        self._where = wheres

    @property
    def _pre_where_(self):
        return self._pre_where

    @_pre_where_.setter
    def _pre_where_(self, pre_wheres):
        self._pre_where = pre_wheres

    @property
    def _group_by_(self):
        return self._group_by

    @_group_by_.setter
    def _group_by_(self, _group_bys):
        self._group_by = _group_bys

    @property
    def _order_by_(self):
        return self._order_by

    @_order_by_.setter
    def _order_by_(self, _order_bys):
        self._order_by = _order_bys

    @property
    def _limit_by_(self):
        return self._limit_by

    @_limit_by_.setter
    def _limit_by_(self, _limit_bys):
        self._limit_by = _limit_bys

    @property
    def _limit_(self):
        return self._limit

    @_limit_.setter
    def _limit_(self, limits):
        self._limit = limits

    @property
    def _having_(self):
        return self._limit

    @_having_.setter
    def _having_(self, limits):
        self._having = limits

    @property
    def table(self):
        return self._table_name

    @table.setter
    def table(self, table_name):
        self._table_name = table_name




class ClickHouseDBNode(LazyInit):  # lazy load to improve loading speed
    def __init__(self, settings: (dict, str) = None):
        super(ClickHouseDBNode, self).__init__()
        if isinstance(settings, dict):
            pass
        elif isinstance(settings, str):
            settings = _parse_rfc1738_args(settings)
        else:
            raise ValueError(f'unknown settings: {settings}')

        self.db_name = settings['database']
        self._settings = settings
        self._conn = ClickHouseBaseNodeQuery(**settings)
        self.is_closed = False
        self._query = self._conn.query
        self._setup_()

    def close(self):
        self._conn.close()
        self.is_closed = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def tables(self):
        sql = 'SHOW TABLES FROM {db}'.format(db=self.db_name)
        res = self._query(sql).values.ravel().tolist()
        return res

    def _setup_(self):
        for table in self.tables:
            if table not in dir(self):
                try:
                    self.__setitem__(table, ClickHouseTableNodeSQLNode(table, **self._settings))
                except Exception as e:
                    warnings.warn(str(e))
                    pass
            else:
                raise ValueError(f'found a table named {table} which is a method or attribute, please rename it')

    def __getitem__(self, table: str):
        return getattr(self, table)

    def __setitem__(self, table: str, table_obj):
        setattr(self, table, table_obj)


if __name__ == '__main__':
    conn_settings = "clickHouse://default:Imsn0wfree@47.105.169.157:8123/NASR_local"

    db = ClickHouseTableNodeSQLNode('NASR_body', settings=conn_settings)
    print(db.query("select * from NASR_local.NASR_body limit 10"))

    print(1)
    # test = [[1, 2, 3], ['str', 'int', 'std']]
    # test_df = pd.DataFrame(test, index=['int', 'str']).T
    # test_df['dt'] = pd.datetime.now()
    # test_df['int'] = test_df['int'].astype(int)
    # test_df['int2'] = test_df['int'].astype(int)
    # test_df['str'] = test_df['str'].astype(str)
    # test_df['str2'] = test_df['str'].astype(str)
    # test_df['category'] = test_df['str'].astype('category')
    # c = DBUtilsCreateTool.translate_dtypes_from_df(test_df)
    # d = test_df.select_dtypes(include=['int'])

    pass
