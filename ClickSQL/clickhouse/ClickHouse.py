# coding=utf-8
import asyncio
import gzip
import json
from collections import namedtuple
from urllib import parse
import pandas as pd
import nest_asyncio
import requests
from aiohttp import ClientSession
from ClickSQL.conf.parse_rfc_1738_args import parse_rfc1738_args
from ClickSQL.utils.file_cache import file_cache

"""
this will hold base function of clickhouse and it will apply a path of access clickhouse through clickhouse api service
this scripts will use none of clickhouse client and only depend on requests to make transactions with 
clickhouse-server

"""

nest_asyncio.apply()  # allow run at jupyter and asyncio env

node_parameters = ['host', 'port', 'user', 'password', 'database']
node = namedtuple('clickhouse', node_parameters)
available_queries_select = ('select', 'show', 'desc')
available_queries_insert = ('insert', 'optimize', 'create')
PRINT_TEST_RESULT = True
SEMAPHORE = 10  # control async number for whole query list


class ParameterKeyError(Exception):
    pass


class ParameterTypeError(Exception):
    pass


class DatabaseTypeError(Exception):
    pass


# TODO change to queue mode change remove aiohttp depends
class ClickHouseTools(object):
    @staticmethod
    def _transfer_sql_format(sql, convert_to, transfer_sql_format=True):
        """
        provide a method which will translate a standard sql into clickhouse sql with might use format as suffix
        :param sql:
        :param convert_to:
        :param transfer_sql_format:
        :return:
        """
        if transfer_sql_format:
            clickhouse_format = 'JSON' if convert_to is None else 'JSONCompact' if convert_to.lower() == 'dataframe' else convert_to
            query_with_format = (sql.rstrip('; \n\t') + ' format ' + clickhouse_format).replace('\n', ' ').strip(' ')
            return query_with_format
        else:
            return sql

    @staticmethod
    def _load_into_pd(ret_value, convert_to: str = 'dataframe', errors='ignore'):
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
            invalid_setting_keys = list(set(settings.keys()) - set(updated_settings.keys()))
            if len(invalid_setting_keys) > 0:
                raise ValueError('setting "{0}" is invalid, valid settings are: {1}'.format(
                    invalid_setting_keys[0], ', '.join(updated_settings.keys())))

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
        self._check_db_settings(db_settings, available_db_type=[node.__name__])

        self._db = db_settings['database']
        self._para = node(db_settings['host'], db_settings['port'], db_settings['user'],
                          db_settings['password'], db_settings['database'])  # store connection information
        self._connect_url = 'http://{user}:{passwd}@{host}:{port}'.format(user=self._para.user,
                                                                          passwd=self._para.password,
                                                                          host=self._para.host,
                                                                          port=self._para.port)
        self.http_settings = self._merge_settings(None, updated_settings=self._default_settings,
                                                  extra_settings={'user': self._para.user,
                                                                  'password': self._para.password})
        # self._session = ClientSession() # the reason of unclose session client
        # self.max_async_query_once = 5
        # self.is_closed = False

        self._test_connection_("http://{host}:{port}/?".format(host=db_settings['host'], port=int(db_settings['port'])))


    @staticmethod
    def _check_db_settings(db_settings: dict, available_db_type=(node.__name__,)):  # node.__name__ : clickhouse
        """
        it is to check db setting whether is correct!
        :param db_settings:
        :return:
        """

        if isinstance(db_settings, dict):
            if db_settings['name'].lower() not in available_db_type:
                raise DatabaseTypeError(
                    f'database symbol is not accept, now only accept: {",".join(available_db_type)}')
            missing_keys = [key for key in node_parameters if key not in db_settings.keys()]
            # :
            #     missing_keys.append(key)
            # else:
            #     pass
            if len(missing_keys) == 0:
                pass
            else:
                raise ParameterKeyError(f"the following keys are not at settings: {','.join(missing_keys)}")
        else:
            raise ParameterTypeError(f'db_setting must be dict type! but get {type(db_settings)}')

    @staticmethod
    def _test_connection_(_base_url):

        """
        a function to test connection by normal way!

        alter function type into staticmethod
        :return:
        """

        ret_value = requests.get(_base_url)
        if PRINT_TEST_RESULT:
            print('connection test: ', ret_value.text.strip())
        del ret_value



    async def _post(self, url: str, sql: str, session):
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

        status = resp.status
        # reason = resp.reason
        if status != 200:
            raise ValueError(result)
        return result

    async def _compression_switched_request(self, query_with_format: (tuple, list, str), convert_to: str = 'dataframe',
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

    @classmethod
    def _load_into_pd_ext(cls, sql: (str, list, tuple), ret_value, convert_to: str, to_df: bool):
        """
        a way to parse into dataframe
        :param sql:
        :param ret_value:
        :param convert_to:
        :param to_df:
        :return:
        """
        if isinstance(sql, str):
            if to_df or ret_value != b'':
                result = cls._load_into_pd(ret_value, convert_to)
            else:
                result = ret_value
        elif isinstance(sql, (list, tuple)):
            if to_df:
                result = [cls._load_into_pd(s, convert_to) if ret_value != b'' else None for s in ret_value]
            else:
                result = ret_value
        else:
            raise ValueError(f'sql must be str or list or tuple,but get {type(sql)}')
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

        sem = asyncio.Semaphore(SEMAPHORE)  # limit async num
        resp_list = self._compression_switched_request(sql, convert_to=convert_to,
                                                       transfer_sql_format=transfer_sql_format, sem=sem)
        if loop is None:
            loop = asyncio.get_event_loop()  # init loop
        res = loop.run_until_complete(resp_list)
        result = self._load_into_pd_ext(sql, res, convert_to, to_df)

        return result

    def execute(self, *sql, convert_to: str = 'dataframe', loop=None, output_df=True, ):
        """
        execute sql or multi sql

        :param output_df:
        :param sql:
        :param convert_to:
        :param loop:
        :return:
        """
        # TODO change to smart mode, can receive any kind sql combination and handle them
        # detect whether all query are insert process
        insert_process = list(map(lambda x: x.lower().startswith(available_queries_insert), sql))
        # detect whether all query are select process
        select_process = list(map(lambda x: x.lower().startswith(available_queries_select), sql))
        if all(insert_process) is True:
            to_df = transfer_sql_format = False
        elif all(select_process) is True:
            to_df = transfer_sql_format = True
        else:
            # TODO change to smart mode, can receive any kind sql combination and handle them
            raise ValueError(
                'the list of queries must be same type query! currently cannot handle various kind SQL type'
                'combination')


        result = self.__execute__(sql, convert_to=convert_to, transfer_sql_format=transfer_sql_format, loop=loop,
                                  to_df=to_df * output_df)

        return result

    def query(self, *sql: str, loop=None, output_df=True, enable_cache=True, exploit_func=True):
        """
        add enable_cache and exploit_func

        ## TODO require to upgrade
        :param exploit_func:
        :param enable_cache:
        :param output_df:
        :param loop:
        :param sql:
        :return:
        """
        func = file_cache(enable_cache=enable_cache, exploit_func=exploit_func)(self.execute)
        result = func(*sql, convert_to='dataframe', loop=loop, output_df=output_df)
        if len(sql) == 1:
            return result[0]
        else:
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

    @property
    def tables(self):
        """
        show tables
        :return:
        """
        sql = 'SHOW TABLES FROM {db}'.format(db=self._db)
        res = self.execute(sql, convert_to='dataframe').values.ravel().tolist()
        return res

    @property
    def databases(self):
        """
        show databases
        :return:
        """
        sql = 'SHOW DATABASES'
        res = self.execute(sql, convert_to='dataframe').values.ravel().tolist()
        return res

    pass


if __name__ == '__main__':
    pass
