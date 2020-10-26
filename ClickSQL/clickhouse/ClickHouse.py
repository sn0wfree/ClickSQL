# coding=utf-8
import asyncio
import gzip
import json
from collections import namedtuple
from urllib import parse

import nest_asyncio
import pandas as pd
import requests
from aiohttp import ClientSession

nest_asyncio.apply()  # allow run at jupyter and asyncio env

node_parameters = ['host', 'port', 'user', 'password', 'database']
node = namedtuple('clickhouse', node_parameters)
available_queries_select = ('select', 'show', 'desc')
available_queries_insert = ('insert', 'optimize', 'create')

SEMAPHORE = 10  # control async number for whole query list

from ClickSQL.conf.parse_rfc_1738_args import _parse_rfc1738_args


class ParameterKeyError(Exception): pass


class ParameterTypeError(Exception): pass


class DatabaseTypeError(Exception): pass


class ClickHouseTools(object):
    @staticmethod
    def _transfer_sql_format(sql, convert_to, transfer_sql_format=True):
        if transfer_sql_format:
            clickhouse_format = 'JSON' if convert_to is None else 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
            query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')
            return query_with_format
        else:
            return sql

    @staticmethod
    def _load_into_pd(ret_value, convert_to: str = 'dataframe', errors='ignore'):
        if convert_to.lower() == 'dataframe':
            result_dict = json.loads(ret_value, strict=False)
            meta = result_dict['meta']
            name = map(lambda x: x['name'], meta)
            data = result_dict['data']
            df = pd.DataFrame.from_records(data, columns=list(name))

            for i in meta:
                if i['type'] in ['DateTime', 'Nullable(DateTime)']:
                    df[i['name']] = pd.to_datetime(df[i['name']], errors=errors)
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


class ClickHouseBaseNode(ClickHouseTools):
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
        self._check_db_settings(db_settings)

        self._para = node(db_settings['host'], db_settings['port'], db_settings['user'],
                          db_settings['password'], db_settings['database'])
        self._db = self._para.database

        self._base_url = "http://{host}:{port}/?".format(host=self._para.host, port=int(self._para.port))

        self.http_settings = self._merge_settings(None, updated_settings=self._default_settings)
        self.http_settings.update({'user': self._para.user, 'password': self._para.password})

        # self._session = ClientSession()
        self.max_async_query_once = 5
        self.is_closed = False

        self._test_connection_()

    @staticmethod
    def _check_db_settings(db_settings: dict, available_db_type=[node.__name__]):  # node.__name__ : clickhouse
        """
        it is to check db setting whether is correct!
        :param db_settings:
        :return:
        """

        if isinstance(db_settings, dict):
            if db_settings['name'].lower() not in available_db_type:
                raise DatabaseTypeError('database symbol is not accept, now only accept: {",".join(available_db_type)}')
            missing_keys = []
            for key in node_parameters:
                if key not in db_settings.keys():
                    missing_keys.append(key)
                else:
                    pass
            if len(missing_keys) == 0:
                pass
            else:
                raise ParameterKeyError(f"the following keys are not at settings: {','.join(missing_keys)}")
        else:
            raise ParameterTypeError(f'db_setting must be dict type! but get {type(db_settings)}')

    def _test_connection_(self):
        """
        is to test connection by normal way!
        :return:
        """
        ret_value = requests.get(self._base_url)
        print('connection test: ', ret_value.text.strip())

    @property
    def _connect_url(self):
        """
        property for base connect
        :return:
        """
        url_str = 'http://{user}:{passwd}@{host}:{port}'.format(user=self._para.user,
                                                                passwd=self._para.password,
                                                                host=self._para.host,
                                                                port=self._para.port
                                                                )
        return url_str

    async def _post(self, url, sql, session):
        """
        the aysnc way to send post request to the server
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

        status = resp.status
        # reason = resp.reason
        if status != 200:
            raise ValueError(result)
        return result

    async def _compression_switched_request(self, query_with_format: str, convert_to: str = 'dataframe',
                                            transfer_sql_format: bool = True, sem=None):
        """
        the core request operator with compression switch adaptor

        :param query_with_format:
        :param convert_to:
        :param transfer_sql_format:
        :param sem:
        :return:
        """
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
        """
        a way to parse into dataframe
        :param sql:
        :param res:
        :param convert_to:
        :param to_df:
        :return:
        """
        if isinstance(sql, str):
            if to_df:
                result = self._load_into_pd(res, convert_to)
            else:
                result = res
        elif isinstance(sql, (list, tuple)):
            if to_df:
                result = [self._load_into_pd(s, convert_to) for s in res]
            else:
                result = res
        else:
            raise ValueError('sql must be str or list or tuple')
        return result

    def __execute__(self, sql: (str, list, tuple), convert_to: str = 'dataframe', transfer_sql_format: bool = True,
                    loop=None, to_df=True):
        """
        the core execute function to run the whole requests and SQL or a list of SQL.
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

    def query(self, *sql: str):
        """
        require to upgrade
        :param sql:
        :return:
        """
        result = self.execute(*sql, convert_to='dataframe', loop=None, )
        return result


class ClickHouseTableNode(ClickHouseBaseNode):
    def __init__(self, conn_str: (str, dict)):
        if isinstance(conn_str, str):
            db_settings = _parse_rfc1738_args(conn_str)
        elif isinstance(conn_str, dict):
            db_settings = conn_str
        else:
            raise ParameterTypeError(f'conn_str must be str or dict but get: {type(conn_str)}')
        super(ClickHouseTableNode, self).__init__(**db_settings)

    @property
    def tables(self):
        sql = 'SHOW TABLES FROM {db}'.format(db=self._db)
        res = self.execute(sql, convert_to='dataframe').values.ravel().tolist()
        return res

    @property
    def databases(self):
        sql = 'SHOW DATABASES'
        res = self.execute(sql, convert_to='dataframe').values.ravel().tolist()
        return res

    pass


if __name__ == '__main__':
    pass
