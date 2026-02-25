# coding=utf-8
import asyncio
import gzip
import warnings
from functools import partial
from urllib import parse
from urllib3.poolmanager import PoolManager

from ClickSQL.clickhouse.ClickHouse import GLOBAL_RAISE_ERROR, ENGAGE_ASYNC, SEMAPHORE, DEFAULT_CONNECT_SETTINGS
from ClickSQL.clickhouse.ClickHouseExt import ClickHouseTableNodeExt
from ClickSQL.errors import ServerError, DatabaseError


class PooledClickHouseTableNodeExt(ClickHouseTableNodeExt):

    def __init__(self, *args, num_pools=2, **kwargs):
        self._pool = PoolManager(num_pools=num_pools)
        self._cached_url = None
        super(PooledClickHouseTableNodeExt, self).__init__(*args, **kwargs)

    def _build_url(self, settings: dict = None) -> str:
        """Build URL with optional settings override."""
        if self._cached_url is None or settings is not None:
            if settings:
                merged_settings = {**self.http_settings, **settings}
                url = f'{self._connect_url}/?{parse.urlencode(merged_settings)}'
            else:
                url = f'{self._connect_url}/?{parse.urlencode(self.http_settings)}'
            if settings is None:
                self._cached_url = url
            return url
        return self._cached_url

    def _post(self, url: str, sql: str, session, raise_error: bool = True):
        if self.http_settings['enable_http_compression'] == 1:
            resp = session.request('POST', url, body=gzip.compress(sql.encode()),
                                   headers={'Content-Encoding': 'gzip',
                                            'Accept-Encoding': 'gzip'})
            result = resp.data
        else:
            resp = session.request('POST', url, body=sql.encode(), )
            result = resp.data

        if resp.status != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise ServerError(result)
            else:
                warnings.warn(str(result))
        return result

    async def _post_async(self, url: str, sql: str, session, raise_error: bool = True):
        """the async way to send post request to the server"""
        if self.http_settings['enable_http_compression'] == 1:
            async with session.post(url, data=gzip.compress(sql.encode()),
                                   headers={'Content-Encoding': 'gzip',
                                            'Accept-Encoding': 'gzip'}) as resp:
                result = await resp.read()
        else:
            async with session.post(url, body=sql.encode()) as resp:
                result = await resp.read()

        if resp.status != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise DatabaseError(result)
            else:
                warnings.warn(str(result))
        return result

    def _compression_switched_request(self, query_with_format: (tuple, list, str), convert_to: str = 'dataframe',
                                      transfer_sql_format: bool = True, sem=None, raise_error=True, settings: dict = None):
        url = self._build_url(settings)
        transfer_sql = partial(self._transfer_sql_format, convert_to=convert_to,
                               transfer_sql_format=transfer_sql_format)

        if isinstance(query_with_format, str):
            result = self._post(url, transfer_sql(query_with_format), self._pool,
                                raise_error=raise_error)
        elif isinstance(query_with_format, (tuple, list)):
            result = [self._post(url, transfer_sql(sql), self._pool, raise_error=raise_error) for sql
                      in query_with_format]
        else:
            raise ValueError('query_with_format must be str , list or tuple')
        return result

    async def _compression_switched_request_async(self, query_with_format: (tuple, list, str),
                                                  convert_to: str = 'dataframe',
                                                  transfer_sql_format: bool = True, sem=None, raise_error=True,
                                                  settings: dict = None):
        """the core request operator with compression switch adaptor"""
        from aiohttp import ClientSession
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

    async def execute_async(self, *sql, convert_to: str = 'dataframe'):
        """Async execute SQL queries"""
        sem = asyncio.Semaphore(SEMAPHORE)
        resp_list = self._compression_switched_request_async(sql, convert_to=convert_to,
                                                              transfer_sql_format=True, sem=sem,
                                                              raise_error=True)
        loop = asyncio.get_event_loop()
        res = await loop.run_until_complete(resp_list)
        result = self._load_into_pd_ext(sql, res, convert_to, to_df=True)
        return result[0] if len(sql) == 1 else result

    def close(self):
        self._pool.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    src = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/system'
    pch = PooledClickHouseTableNodeExt(src)
    for i in range(30):
        c = pch.query('show databases').values.ravel()
        for dd in c:
            c1 = pch.query(f'show tables from {dd}')

    pass
