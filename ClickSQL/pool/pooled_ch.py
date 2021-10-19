# coding=utf-8
import gzip
import warnings
from functools import partial
from urllib import parse
from urllib3.poolmanager import PoolManager

from ClickSQL.clickhouse.ClickHouse import GLOBAL_RAISE_ERROR
from ClickSQL.clickhouse.ClickHouseExt import ClickHouseTableNodeExt
from ClickSQL.errors import ServerError


class PooledClickHouseTableNodeExt(ClickHouseTableNodeExt):

    def __init__(self, *args, num_pools=2, **kwargs):
        self._pool = PoolManager(num_pools=num_pools, )
        super(PooledClickHouseTableNodeExt, self).__init__(*args, **kwargs)

    def _post(self, url: str, sql: str, session, raise_error: bool = True):
        if self.http_settings['enable_http_compression'] == 1:
            resp = session.request('POST', url, body=gzip.compress(sql.encode()),
                                   headers={'Content-Encoding': 'gzip',
                                            'Accept-Encoding': 'gzip'})
            result = resp.data
        else:
            resp = session.request('POST', url, body=sql.encode(), )
            result = resp.data

        # reason = resp.reason
        if resp.status != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise ServerError(result)
            else:
                # result = SmartBytes(result, resp.status_code)
                warnings.warn(str(result))
        return result

    # async def _post_async(self, url: str, sql: str, session, raise_error: bool = True):
    #     """
    #     the async way to send post request to the server
    #     :param url:
    #     :param sql:
    #     :param session:
    #     :return:
    #     """
    #     if self.http_settings['enable_http_compression'] == 1:
    #         resp = session.request('POST', url, body=gzip.compress(sql.encode()),
    #                                headers={'Content-Encoding': 'gzip',
    #                                         'Accept-Encoding': 'gzip'})
    #         result = resp.data
    #     else:
    #         resp = session.request('POST', url, body=sql.encode(), )
    #         result = resp.data
    #
    #     # result = SmartBytes(result, resp.status)
    #     # reason = resp.reason
    #     if resp.status != 200:
    #         if raise_error and GLOBAL_RAISE_ERROR:
    #             raise DatabaseError(result)
    #         else:
    #             warnings.warn(str(result))
    #     return result

    def _compression_switched_request(self, query_with_format: (tuple, list, str), convert_to: str = 'dataframe',
                                      transfer_sql_format: bool = True, sem=None, raise_error=True):
        url = self._connect_url + '/?' + parse.urlencode(self.http_settings)
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

    # async def _compression_switched_request_async(self, query_with_format: (tuple, list, str),
    #                                               convert_to: str = 'dataframe',
    #                                               transfer_sql_format: bool = True, sem=None, raise_error=True):
    #     """
    #     the core request operator with compression switch adaptor
    #
    #     :param query_with_format:
    #     :param convert_to:
    #     :param transfer_sql_format:
    #     :param sem:
    #     :return:
    #     """
    #     url = self._connect_url + '/?' + parse.urlencode(self.http_settings)
    #     transfer_sql = partial(self._transfer_sql_format, convert_to=convert_to,
    #                            transfer_sql_format=transfer_sql_format)
    #     if sem is None:
    #         sem = asyncio.Semaphore(SEMAPHORE)  # limit async num
    #     async with sem:  # limit async number
    #         if isinstance(query_with_format, str):
    #             result = await self._post_async(url, transfer_sql(query_with_format), self._pool,
    #                                             raise_error=raise_error)
    #         elif isinstance(query_with_format, (tuple, list)):
    #             result = [await self._post_async(url, transfer_sql(sql), self._pool, raise_error=raise_error) for sql
    #                       in
    #                       query_with_format]
    #         else:
    #             raise ValueError('query_with_format must be str , list or tuple')
    #     return result

    def close(self):
        self._pool.clear()


if __name__ == '__main__':
    src = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/system'
    pch = PooledClickHouseTableNodeExt(src)
    for i in range(30):
        c = pch.query('show databases').values.ravel()
        for dd in c:
            c1 = pch.query(f'show tables from {dd}')

    pass
