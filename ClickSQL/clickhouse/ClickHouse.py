# coding=utf-8
"""
ClickHouse HTTP Client Module.

This module provides a Python client for ClickHouse database via HTTP protocol.
It supports synchronous and asynchronous operations, connection pooling,
query caching, and various ClickHouse-specific features.

Main Classes:
    - ClickHouseBaseNode: Base class for ClickHouse operations
    - ClickHouseTableNode: Main entry point for table operations

Features:
    - HTTP-based communication (no ClickHouse client needed)
    - Synchronous and asynchronous execution
    - Connection pooling with session reuse
    - Query result caching (LRU + TTL)
    - DataFrame insertion support
    - ClickHouse settings support
    - async_insert support

Example:
    >>> from ClickSQL import ClickHouseTableNode
    >>> node = ClickHouseTableNode("clickhouse://user:pass@host:8123/db")
    >>> result = node.query("SELECT * FROM table LIMIT 10")
"""

import asyncio
import gzip
import warnings
from collections import namedtuple
from functools import partial, partialmethod
from urllib import parse

import numpy as np
import pandas as pd
import requests

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json as _json
    HAS_ORJSON = False

from ClickSQL.conf import Config
from ClickSQL.clickhouse.format import ClickHouseHelper
from ClickSQL.clickhouse.connection import get_session_pool, SessionPool
from ClickSQL.errors import ParameterKeyError, ParameterTypeError
from ClickSQL.errors import ParameterKeyError, ParameterTypeError, DatabaseTypeError, DatabaseError, \
    HeartbeatCheckFailure, ClickHouseTableNotExistsError, ServerError
from ClickSQL.utils import cached_property, file_cache, parse_rfc1738_args
from ClickSQL.utils.query_cache import get_query_cache
if Config.get('ENGAGE_ASYNC', default=False):
    try:
        import nest_asyncio
        nest_asyncio.apply()  # allow run at jupyter and asyncio env
        ENGAGE_ASYNC = True
        from aiohttp import ClientSession
    except Exception as e:
        warnings.warn('Cannot run at jupyter or asyncio env! will use normal version instead!')
        ENGAGE_ASYNC = False
        from requests import Session as ClientSession
else:
    ENGAGE_ASYNC = False
    from requests import Session as ClientSession

node_parameters = ('host', 'port', 'user', 'password', 'database')
node = namedtuple('clickhouse', node_parameters)

PRINT_CHECK_RESULT = Config.get('PRINT_CHECK_RESULT', default=True)
GLOBAL_RAISE_ERROR = Config.get('GLOBAL_RAISE_ERROR', default=True)
SEMAPHORE = Config.get('SEMAPHORE', default=10)  # control async number for whole query list

available_queries_select = ('select', 'show', 'desc')
available_queries_insert = ('insert', 'optimize', 'create')
DEFAULT_CONNECT_SETTINGS = {'enable_http_compression': 1, 'send_progress_in_http_headers': 0,
                            'log_queries': 1, 'connect_timeout': 10, 'receive_timeout': 300,
                            'send_timeout': 300, 'output_format_json_quote_64bit_integers': 0,
                            'wait_end_of_query': 0}


def _json_loads(data):
    """ Unified JSON loads function with orjson fallback. """
    return orjson.loads(data) if HAS_ORJSON else _json.loads(data, strict=False)


def _json_dumps(data, ensure_ascii=False):
    """ Unified JSON dumps function with orjson fallback. """
    if HAS_ORJSON:
        return orjson.dumps(data, ensure_ascii=ensure_ascii).decode('utf-8')
    return _json.dumps(data, ensure_ascii=ensure_ascii)


def _merge_settings(settings: (None, dict), updated_settings: (None, dict) = None,
                    extra_settings: (None, dict) = None) -> dict:
    """
    Merge ClickHouse settings with defaults.
    
    :param settings: User settings
    :param updated_settings: Default settings to update
    :param extra_settings: Extra settings to merge
    :return: Merged settings dict
    """
    if updated_settings is None:
        updated_settings = DEFAULT_CONNECT_SETTINGS.copy()
    elif not isinstance(updated_settings, dict):
        raise ParameterTypeError(f'updated_settings must be dict type, but get {type(updated_settings)}')
    
    if settings is not None and isinstance(settings, dict):
        invalid_setting_keys = set(settings.keys()) - set(updated_settings.keys())
        if len(invalid_setting_keys) > 0:
            raise ValueError('setting "{0}" are invalid, valid settings are: {1}'.format(
                ','.join(invalid_setting_keys), ', '.join(updated_settings.keys())))
        updated_settings.update(settings)
    if extra_settings is not None and isinstance(extra_settings, dict):
        updated_settings.update(extra_settings)

    return {k: v * 1 if isinstance(v, bool) else v for k, v in updated_settings.items()}


class ClickHouseBaseNode(ClickHouseHelper):
    __slots__ = ('_db', '_connect_url', '_para', 'http_settings', '_pool_key', '_reuse_session', '_pool_size', '_session_pool')

    def __init__(self, reuse_session: bool = True, pool_size: int = 5, **db_settings):
        """

        accepted_formats = ('DataFrame', 'TabSeparated', 'TabSeparatedRaw', 'TabSeparatedWithNames',
                        'TabSeparatedWithNamesAndTypes', 'CSV', 'CSVWithNames', 'Values', 'Vertical', 'JSON',
                        'JSONCompact', 'JSONEachRow', 'TSKV', 'Pretty', 'PrettyCompact',
                        'PrettyCompactMonoBlock', 'PrettyNoEscapes', 'PrettySpace', 'XML')

        :param db_settings:
        :param reuse_session: Whether to reuse HTTP sessions for better performance
        :param pool_size: Size of the session pool if reuse_session is True
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
        self._reuse_session = reuse_session
        self._pool_size = pool_size
        self._pool_key = f"{self._para.host}:{self._para.port}:{self._para.database}"
        self._session_pool = get_session_pool()
        
        self.http_settings = _merge_settings(None, updated_settings=DEFAULT_CONNECT_SETTINGS,
                                                   # extra_settings={'user': self._para.user,
                                                   #                 'password': self._para.password}
                                                   )
        _base_url = "http://{host}:{port}/?".format(host=self._para.host, port=int(self._para.port))
        self.__heartbeat_test__(_base_url)
        self._heartbeat_test_ = partialmethod(self.__heartbeat_test__, _base_url=_base_url)
        self.cache_query = partialmethod(self.execute, enable_cache=True, exploit_func=True)

    @staticmethod
    def _check_db_settings_(db_settings: dict, available_db_type=(node.__name__,)):
        """
        Validate database settings.
        
        Checks if the provided db_settings dictionary contains all required
        parameters and has valid values.
        
        Args:
            db_settings: Dictionary containing database connection parameters
            available_db_type: List of acceptable database types
            
        Raises:
            DatabaseTypeError: If database type is not acceptableError: If required
            ParameterKey keys are missing
            ParameterTypeError: If db_settings is not a dictionary
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
            raise HeartbeatCheckFailure(f'heartbeat check failure at {_base_url} with status code: {status_code}')
        if PRINT_CHECK_RESULT:
            print('connection test: ', ret_value.text.strip())
        del ret_value

    def _post(self, url: str, sql: str, session, raise_error: bool = True):
        """
        Send synchronous POST request to ClickHouse server.
        
        Args:
            url: Full URL to send request to
            sql: SQL query string to execute
            session: requests.Session object for connection
            raise_error: Whether to raise exception on error
            
        Returns:
            Raw response bytes from ClickHouse server
            
        Raises:
            ServerError: If server returns error and raise_error is True
        """
        if self.http_settings['enable_http_compression'] == 1:
            with session.post(url, data=gzip.compress(sql.encode()),
                              headers={'Content-Encoding': 'gzip',
                                       'Accept-Encoding': 'gzip'}) as resp:
                result = resp.content
        else:
            with session.post(url, body=sql.encode(), ) as resp:
                result = resp.content

        # reason = resp.reason
        if resp.status_code != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise ServerError(result)
            else:
                # result = SmartBytes(result, resp.status_code)
                warnings.warn(str(result))
        return result

    async def _post_async(self, url: str, sql: str, session, raise_error: bool = True):
        """
        Send asynchronous POST request to ClickHouse server.
        
        Args:
            url: Full URL to send request to
            sql: SQL query string to execute
            session: aiohttp.ClientSession object for connection
            raise_error: Whether to raise exception on error
            
        Returns:
            Raw response bytes from ClickHouse server
            
        Raises:
            DatabaseError: If server returns error and raise_error is True
        """
        if self.http_settings['enable_http_compression'] == 1:
            async with session.post(url, data=gzip.compress(sql.encode()),
                                    headers={'Content-Encoding': 'gzip',
                                             'Accept-Encoding': 'gzip'}) as resp:
                result = await resp.read()
        else:
            async with session.post(url, body=sql.encode(), ) as resp:
                result = await resp.read()

        # result = SmartBytes(result, resp.status)
        # reason = resp.reason
        if resp.status_code != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise DatabaseError(result)
            else:
                warnings.warn(str(result))
        return result

    def _build_url(self, settings: dict = None) -> str:
        """
        Build full URL with query parameters for ClickHouse HTTP request.
        
        Args:
            settings: Optional dict of ClickHouse settings to override defaults
            
        Returns:
            Complete URL string with encoded parameters
        """
        if settings:
            merged_settings = {**self.http_settings, **settings}
            return f'{self._connect_url}/?{parse.urlencode(merged_settings)}'
        return f'{self._connect_url}/?{parse.urlencode(self.http_settings)}'

    def _compression_switched_request(self, query_with_format: (tuple, list, str), convert_to: str = 'dataframe',
                                      transfer_sql_format: bool = True, sem=None, raise_error=True, settings: dict = None):
        """
        Send HTTP request to ClickHouse with optional compression.
        
        This method handles both single query and batch queries, supports
        optional gzip compression, and can use connection pooling.
        
        Args:
            query_with_format: SQL query string or list of queries
            convert_to: Output format ('dataframe', 'JSON', etc.)
            transfer_sql_format: Whether to add format clause to SQL
            sem: Optional semaphore for concurrency control
            raise_error: Whether to raise exception on error
            settings: Optional ClickHouse settings dict
            
        Returns:
            Raw response bytes or list of responses
            
        Raises:
            ValueError: If query_with_format type is invalid
        """

        url = self._build_url(settings)
        transfer_sql = partial(self._transfer_sql_format, convert_to=convert_to,
                               transfer_sql_format=transfer_sql_format)
        
        if self._reuse_session:
            session = self._session_pool.get_session(self._pool_key, self._pool_size)
            if isinstance(query_with_format, str):
                result = self._post(url, transfer_sql(query_with_format), session,
                                    raise_error=raise_error)
            elif isinstance(query_with_format, (tuple, list)):
                result = [self._post(url, transfer_sql(sql), session, raise_error=raise_error) for sql
                          in query_with_format]
            else:
                raise ValueError('query_with_format must be str , list or tuple')
        else:
            with ClientSession() as session:
                if isinstance(query_with_format, str):
                    result = self._post(url, transfer_sql(query_with_format), session,
                                        raise_error=raise_error)
                elif isinstance(query_with_format, (tuple, list)):
                    result = [self._post(url, transfer_sql(sql), session, raise_error=raise_error) for sql
                              in query_with_format]
                else:
                    raise ValueError('query_with_format must be str , list or tuple')
        return result

    async def _compression_switched_request_async(self, query_with_format: (tuple, list, str),
                                                  convert_to: str = 'dataframe',
                                                  transfer_sql_format: bool = True, sem=None, raise_error=True,
                                                  settings: dict = None):
        """Async request handler with compression support."""
        url = self._build_url(settings)
        transfer_sql = partial(self._transfer_sql_format, convert_to=convert_to,
                               transfer_sql_format=transfer_sql_format)
        if sem is None:
            sem = asyncio.Semaphore(SEMAPHORE)
        async with sem:
            async with ClientSession() as session:
                if isinstance(query_with_format, str):
                    result = await self._post_async(url, transfer_sql(query_with_format), session,
                                                    raise_error=raise_error)
                elif isinstance(query_with_format, (tuple, list)):
                    result = [await self._post_async(url, transfer_sql(sql), session, raise_error=raise_error) for sql
                              in query_with_format]
                else:
                    raise ValueError('query_with_format must be str , list or tuple')
        return result

    @classmethod
    def _load_into_pd_ext(cls, sql: (str, list, tuple), ret_value: (bytes, list, tuple), convert_to: str,
                          to_df: bool = True):
        """
        Parse ClickHouse response into DataFrame or return raw data.
        
        Args:
            sql: SQL query or list of queries
            ret_value: Raw response bytes or list of responses
            convert_to: Target format ('dataframe', 'JSON', etc.)
            to_df: Whether to convert to DataFrame
            
        Returns:
            Parsed DataFrame(s) or raw response bytes
        """
        if not to_df:
            result = ret_value
        elif isinstance(sql, str):
            # status code has been removed, if status code != 200 will raise error at post func !
            if ret_value != b'':
                result = cls._load_into_pd(ret_value, convert_to)
            else:
                result = ret_value
        elif isinstance(sql, (list, tuple)):
            result = [cls._load_into_pd(s, convert_to) if s != b'' else s for s in ret_value]
        else:
            raise ValueError(f'sql must be str or list or tuple,but get {type(sql)}')
        return result

    def get_describe_table(self, db, table, filter_type: (list, tuple, None) = None):
        """
        Get table schema/structure from ClickHouse.
        
        Args:
            db: Database name
            table: Table name
            filter_type: Column types to exclude (default: MATERIALIZED, ALIAS)
            
        Returns:
            DataFrame containing table column information
        """
        if filter_type is None:
            filter_type = ['MATERIALIZED', 'ALIAS']

        describe_table = self.__execute__(f'describe table {db}.{table}', convert_to='dataframe',
                                          transfer_sql_format=True,
                                           loop=None, to_df=True, raise_error=True)
        return describe_table[
            ~describe_table['default_type'].isin(filter_type)]

    def _prepare_insert_data(self, df: pd.DataFrame, db: str, table: str, chunksize: int = 100000):
        """
        Prepare insert queries from DataFrame. Extracts common logic for both sync and async insert.
        
        :return: list of SQL insert queries
        """
        from ClickSQL.utils.chunk import chunk

        describe_table = self.get_describe_table(db, table)
        dt_col = describe_table[describe_table['type'].isin(('DateTime', 'Nullable(DateTime)'))]['name'].values.ravel()
        
        df_copy = df.copy()
        for i in dt_col:
            df_copy[i] = pd.to_datetime(df_copy[i]).dt.strftime('%Y-%m-%d %H:%M:%S')

        row_count = df_copy.shape[0]
        rows_data = list(self._check_df_and_dump(df_copy, describe_table))

        db_table = f'{db}.{table}'
        if row_count <= chunksize:
            return [f'insert into {db_table} format JSONEachRow \n' + '\n'.join(rows_data)]
        
        return [f'insert into {db_table} format JSONEachRow \n' + '\n'.join(data) 
                for data in chunk(rows_data, chunksize)]

    def insert_df(self, df: pd.DataFrame, db: str, table: str, chunksize: int = 100000,
                  parallel: bool = False, max_workers: int = 4):
        """
        Insert DataFrame into ClickHouse table.

        :param df: DataFrame to insert
        :param db: target database
        :param table: target table
        :param chunksize: rows per chunk
        :param parallel: use parallel insert
        :param max_workers: max parallel workers
        :return:
        """
        import concurrent.futures

        queries = self._prepare_insert_data(df, db, table, chunksize)

        if parallel and len(queries) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.__execute__, [q], transfer_sql_format=False, loop=None, to_df=False, raise_error=True)
                          for q in queries]
                for f in concurrent.futures.as_completed(futures):
                    f.result()
        else:
            self.__execute__(queries, transfer_sql_format=False, loop=None, to_df=False, raise_error=True)

    async def insert_df_async(self, df: pd.DataFrame, db: str, table: str, chunksize: int = 100000,
                              max_concurrent: int = 4):
        """
        Async insert DataFrame into ClickHouse table.

        :param df: DataFrame to insert
        :param db: target database
        :param table: target table
        :param chunksize: rows per chunk
        :param max_concurrent: max concurrent inserts
        :return:
        """
        import asyncio

        queries = self._prepare_insert_data(df, db, table, chunksize)
        
        sem = asyncio.Semaphore(max_concurrent)
        
        async def _insert_one(q):
            async with sem:
                await self._compression_switched_request_async(q, convert_to='dataframe',
                                                               transfer_sql_format=False, sem=sem,
                                                               raise_error=True)

        loop = asyncio.get_event_loop()
        await loop.run_until_complete(asyncio.gather(*[_insert_one(q) for q in queries]))

    def insert_df_async_server(self, df: pd.DataFrame, db: str, table: str, chunksize: int = 100000,
                                wait_for_async: bool = False):
        """
        Insert DataFrame using ClickHouse async_insert (server-side async).
        
        This method sends data to ClickHouse server which handles batching internally.
        Much faster for high-throughput scenarios as client doesn't wait for flush.
        
        :param df: DataFrame to insert
        :param db: target database
        :param table: target table
        :param chunksize: rows per chunk
        :param wait_for_async: If True, wait for async insert to complete. If False, return immediately.
        :return: Query result
        """
        settings = {
            'async_insert': 1,
            'wait_for_async_insert': 1 if wait_for_async else 0
        }
        
        queries = self._prepare_insert_data(df, db, table, chunksize)
        
        if len(queries) == 1:
            return self.__execute__(queries[0], convert_to='dataframe', transfer_sql_format=False, 
                                   loop=None, to_df=False, raise_error=True, settings=settings)
        
        results = []
        for q in queries:
            result = self.__execute__(q, convert_to='dataframe', transfer_sql_format=False,
                                    loop=None, to_df=False, raise_error=True, settings=settings)
            results.append(result)
        return results

    def __execute__(self, sql: (str, list, tuple), convert_to: str = 'dataframe', transfer_sql_format: bool = True,
                    loop=None, to_df: bool = True, raise_error=True, async_mode=True, settings: dict = None):
        """
        the core execute function to run the whole requests and SQL or a list of SQL.
        :param sql: String or list or tuple
        :param convert_to:
        :param transfer_sql_format:
        :param loop:
        :param to_df:
        :param settings: ClickHouse settings dict (e.g., {'max_threads': 8, 'async_insert': 1})
        :return:
        """
        if async_mode and ENGAGE_ASYNC:
            sem = asyncio.Semaphore(SEMAPHORE)  # limit async num
            resp_list = self._compression_switched_request_async(sql, convert_to=convert_to,
                                                                 transfer_sql_format=transfer_sql_format, sem=sem,
                                                                 raise_error=raise_error, settings=settings)
            if loop is None:
                loop = asyncio.get_event_loop()  # init loop
            res = loop.run_until_complete(resp_list)
        else:
            res = self._compression_switched_request(sql, convert_to=convert_to,
                                                     transfer_sql_format=transfer_sql_format, sem=None,
                                                     raise_error=raise_error, settings=settings)
        result = self._load_into_pd_ext(sql, res, convert_to, to_df=to_df)
        return result

    def execute(self, *sql, convert_to: str = 'dataframe', loop=None, output_df: bool = True,
                enable_cache: bool = False, exploit_func: bool = True, raise_error: bool = True, async_mode=True,
                enable_memory_cache: bool = False, settings: dict = None):
        """
        execute sql or multi sql

        :param async_mode:
        :param raise_error:
        :param exploit_func:
        :param enable_cache:
        :param output_df:
        :param sql:
        :param convert_to:
        :param loop:
        :param enable_memory_cache: use in-memory LRU cache
        :param settings: ClickHouse settings dict (e.g., {'max_threads': 8, 'async_insert': 1})
        :return:
        """
        _query_cache = get_query_cache() if enable_memory_cache else None

        if enable_memory_cache and len(sql) == 1 and isinstance(sql[0], str):
            cached = _query_cache.get(sql[0])
            if cached is not None:
                return cached

        func = file_cache(enable_cache=enable_cache, exploit_func=exploit_func)(self.__execute__)
        result = func(sql, convert_to=convert_to, transfer_sql_format=True, loop=loop,
                      to_df=True * output_df, raise_error=raise_error, async_mode=async_mode, settings=settings)
        final_result = result[0] if len(sql) == 1 else result

        if enable_memory_cache and len(sql) == 1 and isinstance(sql[0], str):
            _query_cache.set(sql[0], final_result)

        return final_result

    def __call__(self, *args, **kwargs):
        """
        Shorthand for query method.
        
        Allows calling node("SELECT ...") directly instead of node.query("SELECT ...")
        
        Args:
            *args: Positional arguments passed to query
            **kwargs: Keyword arguments passed to query
            
        Returns:
            Query result as DataFrame
        """
        return self.query(*args, **kwargs)

    def query(self, *sql: str, loop=None, output_df: bool = True, raise_error=True, async_mode=True, settings: dict = None):

        """
        add enable_cache and exploit_func

        ## TODO require to upgrade
        :param async_mode:
        :param raise_error:
        :param output_df:
        :param loop:
        :param sql:
        :param settings: ClickHouse settings dict (e.g., {'max_threads': 8, 'async_insert': 1})
        :return:
        """

        result = self.execute(*sql, convert_to='dataframe', loop=loop, output_df=output_df, enable_cache=False,
                              exploit_func=False, raise_error=raise_error, async_mode=async_mode, settings=settings)
        return result

    def executemany(self, queries: list, max_workers: int = 4, return_results: bool = True):
        """
        Execute multiple queries in parallel.
        
        :param queries: List of SQL queries
        :param max_workers: Max parallel workers
        :param return_results: Whether to return results
        :return: List of results if return_results=True
        """
        import concurrent.futures
        
        if len(queries) == 1:
            result = self.__execute__(queries[0], transfer_sql_format=True, loop=None, to_df=True, raise_error=True)
            return [result] if return_results else None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.__execute__, q, True, None, True, True): i 
                      for i, q in enumerate(queries)}
            
            results = [None] * len(queries)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                if return_results:
                    results[idx] = future.result()
        
        return results if return_results else None

    async def executemany_async(self, queries: list, max_concurrent: int = 4):
        """
        Execute multiple queries asynchronously.
        
        :param queries: List of SQL queries
        :param max_concurrent: Max concurrent queries
        :return: List of results
        """
        import asyncio
        
        sem = asyncio.Semaphore(max_concurrent)
        
        async def _execute_one(q):
            async with sem:
                return await self._compression_switched_request_async(q, convert_to='dataframe',
                                                                     transfer_sql_format=True, sem=sem,
                                                                     raise_error=True)
        
        loop = asyncio.get_event_loop()
        results = await loop.gather(*[_execute_one(q) for q in queries])
        return [self._load_into_pd(r, 'dataframe') if r else None for r in results]


class ClickHouseTableNode(ClickHouseBaseNode):
    __slots__ = ('_db', '_connect_url', '_para', 'http_settings', '_table', '_pool_key', '_reuse_session', '_pool_size', '_session_pool')

    @staticmethod
    def __parse_conn_str__(conn_str: (str, dict, None) = None, **kwargs):
        """
        Parse connection string or dict into database settings.
        
        Args:
            conn_str: Connection string (e.g., clickhouse://user:pass@host:port/db) or dict
            **kwargs: Alternative key-value pairs for connection
            
        Returns:
            Dictionary with connection parameters
            
        Raises:
            ParameterTypeError: If conn_str type is invalid
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
        return db_settings

    def __init__(self, conn_str: (str, dict, None) = None, reuse_session: bool = True, pool_size: int = 5, **kwargs):
        """
        add kwargs to contain db settings
        :param conn_str:
        :param kwargs:
        :param reuse_session: Whether to reuse HTTP sessions for better performance
        :param pool_size: Size of the session pool
        """
        db_settings = self.__parse_conn_str__(conn_str=conn_str, **kwargs)
        if db_settings['port'] is None:  # add default port for clickhouse
            db_settings['port'] = 8123
        super(ClickHouseTableNode, self).__init__(reuse_session=reuse_session, pool_size=pool_size, **db_settings)
        self._table = self.tables[0]

    @property
    def columns(self):
        """
        Get list of column names for current table.
        
        Returns:
            List of column names
        """
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
        Get list of table names in current database.
        
        Returns:
            List of table names
        """
        sql = 'SHOW TABLES FROM {db}'.format(db=self._db)
        res = self.execute(sql, convert_to='dataframe').values.ravel().tolist()
        return res

    @cached_property
    def databases(self):
        """
        Get list of all databases on ClickHouse server.
        
        Returns:
            List of database names (cached property)
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
        Check if a database or table exists.
        
        Args:
            db_table: Table name (db.table format) or just table name
            mode: 'table' or 'database'
            output: Whether to print detection message
            
        Returns:
            True if exists, False otherwise
            
        Raises:
            ValueError: If db_table format is invalid
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
        else:
            return db in self.databases


if __name__ == '__main__':
    pass
