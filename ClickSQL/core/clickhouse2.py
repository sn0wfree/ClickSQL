# coding=utf-8
import gzip
from collections import ChainMap, namedtuple
from urllib import parse

import asyncio
import json
import pandas as pd
from aiohttp import ClientSession

node = namedtuple('clickhouse', ['host', 'port', 'user', 'password', 'database'])


class ClickhouseTools(object):
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


class ClickhouseBaseNode(ClickhouseTools):
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

        # self._test_connection_()

    @property
    def _connect_url(self):
        url_str = 'http://{user}:{passwd}@{host}:{port}'.format(user=self._para.user,
                                                                passwd=self._para.password,
                                                                host=self._para.host,
                                                                port=self._para.port
                                                                )
        return url_str

    async def _compression_switched_request(self, query_with_format: str, convert_to: str = 'dataframe',
                                            transfer_sql_format: bool = True):
        sql2 = self._transfer_sql_format(query_with_format, convert_to=convert_to,
                                         transfer_sql_format=transfer_sql_format)
        url = self._connect_url + '/?' + parse.urlencode(self.http_settings)
        async with ClientSession() as session:
            if self.http_settings['enable_http_compression'] == 1:
                async with session.post(url, data=gzip.compress(sql2.encode()),
                                        headers={'Content-Encoding': 'gzip',
                                                 'Accept-Encoding': 'gzip'}) as resp:
                    result = await resp.read()
            else:
                async with session.post(url, body=sql2.encode(), ) as resp:
                    result = await resp.read()

            status = resp.status
            reason = resp.reason
        return result, status, reason

    def __execute__(self, sql: str, convert_to: str = 'dataframe', transfer_sql_format: bool = True, loop=None):
        resp_list = self._compression_switched_request(sql, convert_to=convert_to,
                                                       transfer_sql_format=transfer_sql_format)

        if loop is None:
            loop = asyncio.get_event_loop()

        result, status, reason = loop.run_until_complete(resp_list)
        if status != 200:
            raise ValueError(reason)

        return result

    def execute(self, sql: str, convert_to: str = 'dataframe', transfer_sql_format: bool = True, loop=None, ):
        if sql.lower().startswith(('insert', 'optim', 'create')):
            to_df = False
        else:
            to_df = True

        result = self.__execute__(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format, loop=loop)
        if to_df:
            d = self._load_into_pd(result, convert_to)
            return d
        else:
            pass

    async def __request_async_unit__(self, sql: str, convert_to: str = 'dataframe', transfer_sql_format: bool = True):
        sql2 = self._transfer_sql_format(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format)

        if self.http_settings['enable_http_compression'] == 1:
            url = self._base_url + parse.urlencode(self.http_settings)
            resp = await self._session._request('POST', url,
                                                data=gzip.compress(sql2.encode()),
                                                headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'})
        else:
            # settings = self.http_settings.copy()
            # settings.update({'query': sql2})
            url = self._base_url + parse.urlencode(ChainMap(self.http_settings, {'query': sql2}))
            resp = await self._session._request('POST', url)
        return resp


if __name__ == '__main__':
    node = ClickhouseBaseNode(
        **{'host': '47.104.186.157', 'port': 8123, 'user': 'default', 'password': 'Imsn0wfree',
           'database': 'EDGAR_LOG'})
    df = node.execute('show tables from system')
    print(df)
    pass
