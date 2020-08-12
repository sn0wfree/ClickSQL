# coding=utf-8
import asyncio
import gzip
import json
import re
from collections import namedtuple
from urllib import parse

import nest_asyncio
import pandas as pd
import requests
from aiohttp import ClientSession

nest_asyncio.apply()
node = namedtuple('clickhouse', ['host', 'port', 'user', 'password', 'database'])
available_queries_select = ('select', 'show', 'desc')
available_queries_insert = ('insert', 'optimize', 'create')

SEMAPHORE = 10


class ClickHouseCreateTableTools(object):
    @classmethod
    def _create_table_sql(cls, db: str, table: str, dtypes_dict: dict,
                          order_by_key_cols: (list, tuple),
                          primary_key_cols=None, sample_expr=None,
                          engine_type: str = 'ReplacingMergeTree', partitions_expr=None,
                          settings="SETTINGS index_granularity = 8192", other=''):
        """

        :param db:
        :param table:
        :param dtypes_dict:
        :param order_by_key_cols:
        :param primary_key_cols:
        :param sample_expr:
        :param engine_type:
        :param partitions_expr:
        :param settings:
        :param other:
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

        cols_def = ','.join([f"{name} {d_type}" for name, d_type in dtypes_dict.items()])

        maid_body = f"CREATE TABLE IF NOT EXISTS {db}.{table} ( {cols_def} ) ENGINE = {engine_type}"

        ORDER_BY_CLAUSE = f"ORDER BY ( {','.join(order_by_key_cols)} )"

        if partitions_expr is None:
            PARTITION_by_CLAUSE = ''
        else:
            PARTITION_by_CLAUSE = f"PARTITION BY {partitions_expr}"

        if primary_key_cols is None:
            PRIMARY_BY_CLAUSE = ''
        else:
            primary_key_expr = ','.join(primary_key_cols)
            PRIMARY_BY_CLAUSE = f"PARTITION BY ({primary_key_expr})"

        if sample_expr is None:
            SAMPLE_CLAUSE = ''
        else:
            SAMPLE_CLAUSE = sample_expr

        base = f"{maid_body} {PARTITION_by_CLAUSE} {ORDER_BY_CLAUSE} {PRIMARY_BY_CLAUSE} {SAMPLE_CLAUSE} {other} {settings}"

        return base

    @staticmethod
    def _check_end_with_limit(string, pattern=r'[\s]+limit[\s]+[0-9]+$'):
        m = re.findall(pattern, string)
        if m is None or m == []:
            return False
        else:
            return True

    @classmethod
    def _create_table_from_sql(cls, db: str, table: str, sql: str, key_cols: list,
                               extra_format_dict: (dict, None) = None,
                               primary_key_cols=None, sample_expr=None, other='',
                               engine_type: str = 'ReplacingMergeTree',
                               partitions_expr=None, query_func=None):
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

        if isinstance(sql, str):
            pass
        else:
            raise ValueError('sql must be string')

        limit_status = cls._check_end_with_limit(sql, pattern=r'[\s]+limit[\s]+[0-9]+$')
        if limit_status:
            describe_sql = f' describe({sql}) '
        else:
            describe_sql = f'describe ( {sql} limit 1)'

        if query_func is None:
            raise ValueError('query function should be set!')

        dtypes_df = query_func(describe_sql)

        dtypes_dict = dict(dtypes_df[['name', 'type']].drop_duplicates().values)
        if extra_format_dict is None:
            pass
        else:
            dtypes_dict.update(extra_format_dict)
        sql = cls._create_table_sql(db, table, dtypes_dict, key_cols, engine_type=engine_type,
                                    primary_key_cols=primary_key_cols, sample_expr=sample_expr,
                                    partitions_expr=partitions_expr, other=other)

        return sql

    @staticmethod
    def translate_dtypes1_as_dtypes2(df: pd.DataFrame, src2target={'category': 'str'}):
        dtypes_series = df.dtypes
        for src, dest in src2target.items():
            if src in dtypes_series:
                category_cols = dtypes_series[dtypes_series == src].index
                for col in category_cols:
                    df[col] = df[col].astype(dest)
            else:
                pass
        return df

    @staticmethod
    def translate_dtypes_from_df(df: pd.DataFrame, translate_dtypes: dict = {'object': 'String',
                                                                             'datetime64[ns]': 'Datetime'}):
        if hasattr(df, 'dtypes'):
            dtypes_series = df.dtypes.replace(translate_dtypes)
            return dtypes_series.map(lambda x: str(x).capitalize()).to_dict()
        elif hasattr(df, '_columns_') and 'type' in df._columns_ and 'name' in df._columns_:
            dtypes_series = df.set_index('name')['type'].replace(translate_dtypes)
            return dtypes_series.map(lambda x: str(x)).to_dict()
        else:
            raise ValueError(f'unknown df:{type(df)}')

    @classmethod
    def _create_table_from_df(cls, db: str, table: str, df: pd.DataFrame, key_cols: (list, tuple),
                              engine_type: str = 'ReplacingMergeTree', extra_format_dict=None, partitions_expr=None,
                              src2target={'category': 'str'},
                              query_func=None
                              ):

        df = cls.translate_dtypes1_as_dtypes2(df, src2target={'category': 'str'})
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


class ClickHouseTools(ClickHouseCreateTableTools):

    @classmethod
    def _create_table_from_df(cls, obj: object, db: str, table: str, df: pd.DataFrame, key_cols: (list, tuple),
                              engine_type: str = 'ReplacingMergeTree', extra_format_dict=None, partitions_expr=None):
        query_func = obj.query

        df = cls.translate_dtypes1_as_dtypes2(df, src2target={'category': 'str'})
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
                    df[i['name']] = pd.to_datetime(df[i['name']], errors='ignore')
            ret_value = df
        return ret_value

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


class ClickhouseBaseNode(ClickHouseTools):
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

        self._para = node(db_settings['host'], db_settings['port'], db_settings['user'],
                          db_settings['password'], db_settings['database'])

        self._base_url = "http://{host}:{port}/?".format(host=self._para.host, port=int(self._para.port))

        self.http_settings = self._merge_settings(None, updated_settings=self._default_settings)
        self.http_settings.update({'user': self._para.user, 'password': self._para.password})

        self._session = ClientSession()
        self.max_async_query_once = 5
        self.is_closed = False

        self._test_connection_()

    def _test_connection_(self):
        ret_value = requests.get(self._base_url)
        print('connection test: ', ret_value.text.strip())

    @property
    def _connect_url(self):
        url_str = 'http://{user}:{passwd}@{host}:{port}'.format(user=self._para.user,
                                                                passwd=self._para.password,
                                                                host=self._para.host,
                                                                port=self._para.port
                                                                )
        return url_str

    async def _post(self, url, sql, session):
        if self.http_settings['enable_http_compression'] == 1:
            async with session.post(url, data=gzip.compress(sql.encode()),
                                    headers={'Content-Encoding': 'gzip',
                                             'Accept-Encoding': 'gzip'}) as resp:
                result = await resp.read()
        else:
            async with session.post(url, body=sql.encode(), ) as resp:
                result = await resp.read()

        status = resp.status
        reason = resp.reason
        if status != 200:
            raise ValueError(result)
        return result

    async def _compression_switched_request(self, query_with_format: str, convert_to: str = 'dataframe',
                                            transfer_sql_format: bool = True, sem=None):
        url = self._connect_url + '/?' + parse.urlencode(self.http_settings)
        if sem is None:
            sem = asyncio.Semaphore(SEMAPHORE)  # limit async num
        async with sem:  # limit async number
            async with ClientSession() as session:
                if isinstance(query_with_format, str):
                    sql2 = self._transfer_sql_format(query_with_format, convert_to=convert_to,
                                                     transfer_sql_format=transfer_sql_format)
                    result = await self._post(url, sql2, session)
                elif isinstance(query_with_format, (tuple, list)):
                    result = []
                    for sql in query_with_format:
                        s = self._transfer_sql_format(sql, convert_to=convert_to,
                                                      transfer_sql_format=transfer_sql_format)
                        res = await self._post(url, s, session)
                        result.append(res)
                else:
                    raise ValueError('query_with_format must be str , list or tuple')

        return result

    def _load_into_pd_ext(self, sql, res, convert_to, to_df):
        if isinstance(sql, str):
            if to_df:
                result = self._load_into_pd(res, convert_to)
        elif isinstance(sql, (list, tuple)):
            if to_df:
                result = [self._load_into_pd(s, convert_to) for s in res]
        else:
            raise ValueError('sql must be str or list or tuple')
        return result

    def __execute__(self, sql: (str, list, tuple), convert_to: str = 'dataframe', transfer_sql_format: bool = True,
                    loop=None, to_df=True):
        """

        :param sql:
        :param convert_to:
        :param transfer_sql_format:
        :param loop:
        :param to_df:
        :return:
        """

        if loop is None:
            loop = asyncio.get_event_loop()

        sem = asyncio.Semaphore(SEMAPHORE)  # limit async num

        resp_list = self._compression_switched_request(sql, convert_to=convert_to,
                                                       transfer_sql_format=transfer_sql_format, sem=sem)

        res = loop.run_until_complete(resp_list)

        result = self._load_into_pd_ext(sql, res, convert_to, to_df)

        return result

    def execute(self, *sql, convert_to: str = 'dataframe', loop=None, ):
        """

        :param sql:
        :param convert_to:
        :param loop:
        :return:
        """
        ## TODO warning: Unclosed client session

        insert_process = list(map(lambda x: x.lower().startswith(available_queries_insert), sql))
        select_process = list(map(lambda x: x.lower().startswith(available_queries_select), sql))
        if all(insert_process) is True:
            to_df = False
            transfer_sql_format = False
        elif all(select_process) is True:
            to_df = True
            transfer_sql_format = True
        else:
            raise ValueError('the list of query must be same type query!')

        if len(sql) != 1:

            result = self.__execute__(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format, loop=loop,
                                      to_df=to_df)
        else:
            result = self.__execute__(sql[0], convert_to=convert_to, transfer_sql_format=transfer_sql_format, loop=loop,
                                      to_df=to_df)
        return result

    def query(self, sql: str):
        result = self.execute(sql, convert_to='dataframe', loop=None, )
        return result


class ClickHouseTableNode(ClickhouseBaseNode):
    def __init__(self, **db_settings):
        super(ClickHouseTableNode, self).__init__(**db_settings)

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

    pass


if __name__ == '__main__':
    node = ClickHouseTableNode(
        **{'host': '47.104.186.157', 'port': 8123, 'user': 'default', 'password': 'Imsn0wfree',
           'database': 'EDGAR_LOG'})
    df1 = node.tables
    dff = node.query("select * from system.parts")
    pass
