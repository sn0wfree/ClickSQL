# coding=utf-8
import gzip
import warnings
from collections import namedtuple
from urllib import parse

import asyncio
import json
import nest_asyncio
import pandas as pd
import requests
from aiohttp import ClientSession
from functools import partial, partialmethod,cached_property

from ClickSQL.conf.parse_rfc_1738_args import parse_rfc1738_args
from ClickSQL.errors import ParameterKeyError, ParameterTypeError, DatabaseTypeError, DatabaseError, \
    HeartbeatCheckFailure, ClickHouseTableNotExistsError
from ClickSQL.utils.file_cache import file_cache

"""
this will hold base function of clickhouse and it will apply a path of access clickhouse through clickhouse api service
this scripts will use none of clickhouse client and only depend on requests to make transactions with 
clickhouse-server

"""

nest_asyncio.apply()  # allow run at jupyter and asyncio env

node_parameters = ('host', 'port', 'user', 'password', 'database')
node = namedtuple('clickhouse', node_parameters)
available_queries_select = ('select', 'show', 'desc')
available_queries_insert = ('insert', 'optimize', 'create')
PRINT_CHECK_RESULT = True
GLOBAL_RAISE_ERROR = True
SEMAPHORE = 10  # control async number for whole query list


# class SmartResult(object):
#
#     def __init__(self, result, status_code: int):
#         self._obj = result
#         self.status_code = status_code


def SmartBytes(result: bytes, status_code: int):
    result_cls = type('SmartBytes', (bytes,), {'status_code': property(lambda x: status_code)})

    return result_cls(result)


# TODO change to queue mode change remove aiohttp depends
class ClickHouseTools(object):
    @staticmethod
    def _check_sql_type(sql: str):
        if sql.lower().startswith(available_queries_select):
            return 'select-liked'
        elif sql.lower().startswith(available_queries_insert):
            return 'insert-liked'

        else:
            raise ValueError('sql type cannot be supported! need upgrade!')

    @staticmethod
    def _transfer_sql_format(sql: str, convert_to: str, transfer_sql_format: bool = True):
        """
        provide a method which will translate a standard sql into clickhouse sql with might use format as suffix
        :param sql:
        :param convert_to:
        :param transfer_sql_format:
        :return:
        """
        if transfer_sql_format:
            if convert_to is None:
                clickhouse_format = 'JSON'
            else:
                clickhouse_format = 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
            query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')
            return query_with_format
        else:
            return sql

    @staticmethod
    def _load_into_pd(ret_value: (str, bytes), convert_to: str = 'dataframe', errors='ignore'):
        """
        will provide a approach to load data from clickhouse into pd.DataFrame format which may be easy to use

        :param ret_value:
        :param convert_to:
        :param errors:
        :return:
        """

        if convert_to.lower() == 'dataframe':
            result_dict = json.loads(ret_value, strict=False)
            meta = result_dict['meta']
            name = map(lambda x: x['name'], meta)
            df = pd.DataFrame.from_records(result_dict['data'], columns=list(name))
            for i in meta:
                if i['type'] in ['DateTime', 'Nullable(DateTime)']:  # translate format
                    # process datetime format
                    df[i['name']] = pd.to_datetime(df[i['name']], errors=errors)
            return df
        else:
            return ret_value

    @classmethod
    def _merge_settings(cls, settings: (None, dict), updated_settings: (None, dict) = None,
                        extra_settings: (None, dict) = None):
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
        elif not isinstance(updated_settings, dict):
            raise ParameterTypeError(f'updated_settings must be dict type, but get {type(updated_settings)}')
        else:
            pass
        if settings is not None and isinstance(settings, dict):
            invalid_setting_keys = set(settings.keys()) - set(updated_settings.keys())
            if len(invalid_setting_keys) > 0:
                raise ValueError('setting "{0}" are invalid, valid settings are: {1}'.format(
                    ','.join(invalid_setting_keys), ', '.join(updated_settings.keys())))
            updated_settings.update(settings)
        if extra_settings is not None and isinstance(extra_settings, dict):
            updated_settings.update(extra_settings)

        return {k: v * 1 if isinstance(v, bool) else v for k, v in updated_settings.items()}


class ClickHouseBaseNode(ClickHouseTools):
    _default_settings = {'enable_http_compression': 1, 'send_progress_in_http_headers': 0,
                         'log_queries': 1, 'connect_timeout': 10, 'receive_timeout': 300,
                         'send_timeout': 300, 'output_format_json_quote_64bit_integers': 0,
                         'wait_end_of_query': 0}

    __slots__ = ('_db', '_connect_url', '_para', 'http_settings')  # 'max_async_query_once', 'is_closed'

    def __init__(self, **db_settings):
        """

        accepted_formats = ('DataFrame', 'TabSeparated', 'TabSeparatedRaw', 'TabSeparatedWithNames',
                        'TabSeparatedWithNamesAndTypes', 'CSV', 'CSVWithNames', 'Values', 'Vertical', 'JSON',
                        'JSONCompact', 'JSONEachRow', 'TSKV', 'Pretty', 'PrettyCompact',
                        'PrettyCompactMonoBlock', 'PrettyNoEscapes', 'PrettySpace', 'XML')

        :param db_settings:
        """
        self._check_db_settings_(db_settings, available_db_type=[node.__name__])
        self._para = node(db_settings['host'], db_settings['port'], db_settings['user'],
                          db_settings['password'], db_settings['database'])  # store connection information
        if '.' in self._para.database:
            self._db = self._para.database.split('.')[0]
        else:
            self._db = self._para.database
        self._connect_url = 'http://{user}:{passwd}@{host}:{port}'.format(user=self._para.user,
                                                                          passwd=self._para.password,
                                                                          host=self._para.host,
                                                                          port=self._para.port)
        self.http_settings = self._merge_settings(None, updated_settings=self._default_settings,
                                                  extra_settings={'user': self._para.user,
                                                                  'password': self._para.password})
        # self.max_async_query_once = 5
        # self.is_closed = False
        _base_url = "http://{host}:{port}/?".format(host=self._para.host, port=int(self._para.port))
        self.__heartbeat_test__(_base_url)
        self._heartbeat_test_ = partialmethod(self.__heartbeat_test__, _base_url=_base_url)
        self.cache_query = partialmethod(self.execute, enable_cache=True, exploit_func=True)

    @staticmethod
    def _check_db_settings_(db_settings: dict, available_db_type=(node.__name__,)):  # node.__name__ : clickhouse
        """
        it is to check db setting whether is correct!
        :param db_settings:
        :return:
        """

        if isinstance(db_settings, dict):
            if db_settings['name'].lower() not in available_db_type:
                raise DatabaseTypeError(
                    f'database symbol is not accepted, now only accept: {",".join(available_db_type)}')

            missing_keys = filter(lambda x: x not in db_settings.keys(), node_parameters)  # can improve
            if len(tuple(missing_keys)) == 0:
                pass
            else:
                raise ParameterKeyError(f"the following keys are not at settings: {','.join(missing_keys)}")
        else:
            raise ParameterTypeError(f'db_setting must be dict type! but get {type(db_settings)}')

    @staticmethod
    def __heartbeat_test__(_base_url: str):
        """
        a function to test connection by normal way!

        alter function type into staticmethod
        :return:
        """

        ret_value = requests.get(_base_url)
        status_code = ret_value.status_code
        if status_code != 200:
            raise HeartbeatCheckFailure(f'heartbeat check failure at {_base_url}')
        if PRINT_CHECK_RESULT:
            print('connection test: ', ret_value.text.strip())
        del ret_value

    async def _post(self, url: str, sql: str, session, raise_error: bool = True):
        """
        the async way to send post request to the server
        :param url:
        :param sql:
        :param session:
        :return:
        """
        if self.http_settings['enable_http_compression'] == 1:
            async with session.post(url, data=gzip.compress(sql.encode()),
                                    headers={'Content-Encoding': 'gzip',
                                             'Accept-Encoding': 'gzip'}) as resp:
                result = await resp.read()
        else:
            async with session.post(url, body=sql.encode(), ) as resp:
                result = await resp.read()

        result = SmartBytes(result, resp.status)
        # reason = resp.reason
        if result.status_code != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise DatabaseError(result)
            else:
                warnings.warn(str(result))
        return result

    async def _compression_switched_request(self, query_with_format: (tuple, list, str), convert_to: str = 'dataframe',
                                            transfer_sql_format: bool = True, sem=None, raise_error=True):
        """
        the core request operator with compression switch adaptor

        :param query_with_format:
        :param convert_to:
        :param transfer_sql_format:
        :param sem:
        :return:
        """
        url = self._connect_url + '/?' + parse.urlencode(self.http_settings)
        transfer_sql = partial(self._transfer_sql_format, convert_to=convert_to,
                               transfer_sql_format=transfer_sql_format)
        if sem is None:
            sem = asyncio.Semaphore(SEMAPHORE)  # limit async num
        async with sem:  # limit async number
            async with ClientSession() as session:
                if isinstance(query_with_format, str):
                    result = await self._post(url, transfer_sql(query_with_format), session, raise_error=raise_error)
                elif isinstance(query_with_format, (tuple, list)):
                    result = [await self._post(url, transfer_sql(sql), session, raise_error=raise_error) for sql in
                              query_with_format]
                else:
                    raise ValueError('query_with_format must be str , list or tuple')
        return result

    @classmethod
    def _load_into_pd_ext(cls, sql: (str, list, tuple), ret_value: (bytes, list, tuple), convert_to: str,
                          to_df: bool = True):
        """
        a way to parse into dataframe
        :param sql:
        :param ret_value:
        :param convert_to:
        :param to_df:
        :return:
        """
        if not to_df:
            result = ret_value
        elif isinstance(sql, str):
            if ret_value != b'' and ret_value.status_code == 200:
                result = cls._load_into_pd(ret_value, convert_to)
            else:
                result = ret_value
        elif isinstance(sql, (list, tuple)):
            result = [cls._load_into_pd(s, convert_to) if s != b'' and s.status_code == 200 else s for s in
                      ret_value]
        else:
            raise ValueError(f'sql must be str or list or tuple,but get {type(sql)}')
        return result

    def __execute__(self, sql: (str, list, tuple), convert_to: str = 'dataframe', transfer_sql_format: bool = True,
                    loop=None, to_df: bool = True, raise_error=True):
        """
        the core execute function to run the whole requests and SQL or a list of SQL.
        :param sql:
        :param convert_to:
        :param transfer_sql_format:
        :param loop:
        :param to_df:
        :return:
        """

        sem = asyncio.Semaphore(SEMAPHORE)  # limit async num
        resp_list = self._compression_switched_request(sql, convert_to=convert_to,
                                                       transfer_sql_format=transfer_sql_format, sem=sem,
                                                       raise_error=raise_error)
        if loop is None:
            loop = asyncio.get_event_loop()  # init loop
        res = loop.run_until_complete(resp_list)
        result = self._load_into_pd_ext(sql, res, convert_to, to_df=to_df)
        return result

    def execute(self, *sql, convert_to: str = 'dataframe', loop=None, output_df: bool = True,
                enable_cache: bool = False, exploit_func: bool = True, raise_error: bool = True):
        """
        execute sql or multi sql

        :param raise_error:
        :param exploit_func:
        :param enable_cache:
        :param output_df:
        :param sql:
        :param convert_to:
        :param loop:
        :return:
        """

        # detect whether all query are insert process
        # insert_process = map(lambda x: x.lower().startswith(available_queries_insert), sql)
        # # detect whether all query are select process
        # select_process = map(lambda x: x.lower().startswith(available_queries_select), sql)
        # if all(list(select_process)) is True:
        #     to_df = transfer_sql_format = True
        # elif all(list(insert_process)) is True:
        #     to_df = transfer_sql_format = False
        # else:
        #
        #     raise ValueError(
        #         'the list of queries must be same type query! currently cannot handle various kind SQL type'
        #         'combination')
        func = file_cache(enable_cache=enable_cache, exploit_func=exploit_func)(self.__execute__)
        result = func(sql, convert_to=convert_to, transfer_sql_format=True, loop=loop,
                      to_df=True * output_df, raise_error=raise_error)

        if len(sql) == 1:
            return result[0]
        else:

            return result

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return self.query(*args, **kwargs)

    def query(self, *sql: str, loop=None, output_df: bool = True, raise_error=True):

        """
        add enable_cache and exploit_func

        ## TODO require to upgrade
        :param raise_error:
        :param output_df:
        :param loop:
        :param sql:
        :return:
        """

        result = self.execute(*sql, convert_to='dataframe', loop=loop, output_df=output_df, enable_cache=False,
                              exploit_func=False, raise_error=raise_error)
        return result


class ClickHouseTableNode(ClickHouseBaseNode):

    def __init__(self, conn_str: (str, dict, None) = None, **kwargs):
        """
        add kwargs to contain db settings
        :param conn_str:
        :param kwargs:
        """
        if conn_str is None:
            if kwargs != {}:
                db_settings = kwargs
                db_settings['name'] = node.__name__
            else:
                raise ParameterTypeError('database parameters cannot be parsed normally!')
        elif isinstance(conn_str, str):
            db_settings = parse_rfc1738_args(conn_str)
        elif isinstance(conn_str, dict):
            db_settings = conn_str
        else:
            raise ParameterTypeError(f'conn_str must be str or dict but get: {type(conn_str)}')

        if db_settings['port'] is None:  # add default port for clickhouse
            db_settings['port'] = 8123
        super(ClickHouseTableNode, self).__init__(**db_settings)
        self._table = self.tables[0]

    @property
    def columns(self):
        # db_table: str = None
        db_table = f"{self._db}.{self._table}"
        sql = f'desc {db_table}'
        res = self.execute(sql, convert_to='dataframe')['name'].tolist()
        return res

    @columns.setter
    def columns(self, table: str):
        if table in self.tables:
            self._table = table
        else:
            raise ClickHouseTableNotExistsError(f'{table} not at {self._db}')

    @property
    def tables(self):
        """
        show tables
        :return:
        """
        sql = 'SHOW TABLES FROM {db}'.format(db=self._db)
        res = self.execute(sql, convert_to='dataframe').values.ravel().tolist()
        return res

    @cached_property
    def databases(self):
        """
        show databases
        :return:
        """
        sql = 'SHOW DATABASES'
        res = self.execute(sql, convert_to='dataframe').values.ravel().tolist()
        return res

    @staticmethod
    def add_auto_increment_materialized_col(db_table: str, fid_col_name: str = 'fid'):
        """
        executable sql for clickhouse to add materialized column with auto-increment
        :param db_table:
        :param fid_col_name:
        :return:
        """

        exec_express = "bitOr(bitShiftLeft(toUInt64(now64()),24), rowNumberInAllBlocks())"
        return f"alter table {db_table} add column {fid_col_name} Int64  MATERIALIZED {exec_express}"

    def _check_exists(self, db_table: str, mode: str = 'table', output=True):
        """

        :param db_table:
        :param mode:
        :return:
        """
        # TODO check table exists
        if isinstance(db_table, str):
            if '.' in db_table:
                db, table = db_table.split('.')
            else:
                db, table = self._db, db_table
        else:
            raise ValueError('please input correct db.table information')
        if output:
            print(f'will detect {db}.{table} existence!')

        if mode == 'table':
            if db == self._db:
                return table in self.tables
            else:
                sql = f"show tables from {db}"
                tables = self.query(sql).values.ravel().tolist()
                return table in tables
            # else:
            # else:
            #     return db in self.databases
        else:

            return db in self.databases


if __name__ == '__main__':
    pass
