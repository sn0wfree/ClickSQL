# coding=utf-8

import json
import pandas as pd
import requests
import time
from functools import partial

from ClickSQL.utils.file_cache import file_cache
from ClickSQL.utils.process_bar import process_bar

Base_URL = 'http://api.waditu.com'
Token = '779d968ebbfab103168884f05edb4ebfb10f269bdd6e235ba3a5fe25'
from ClickSQL import BaseSingleFactorTableNode

args = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/system.tables'
Node = BaseSingleFactorTableNode(args)


# {'api_name'：接口名称，比如stock_basic
#
# token ：用户唯一标识，可通过登录pro网站获取
#
# params：接口参数，如daily接口中start_date和end_date
#
# fields：字段列表，用于接口获取指定的字段，以逗号分隔，如"open,high,low,close"}


def to_df(result):
    if isinstance(result, dict):
        cols = result['fields']
        return pd.DataFrame(result['items'], columns=cols)
    else:
        return result


@file_cache()
def _get_req(fields, api_name, token, **params):
    token = Token if token is None else token
    req_params = {'api_name': api_name, 'token': token, 'params': params, "fields": fields}
    resp = requests.post(Base_URL, json.dumps(req_params).encode('utf-8'), )
    if resp.status_code == 200:
        result = json.loads(resp.content.decode('utf-8'))
        if result['code'] != 0:
            raise Exception(result['msg'])

        return to_df(result['data'])
    else:
        raise ValueError(f'{resp.text}')


def auto_create(db, table, key_cols, engine_type='MergeTree'):
    def _auto_create(func):
        def _c(*args, **kwargs):
            df = func(*args, **kwargs)
            Node.create(db,
                        table,
                        df_or_sql_or_dict=df,
                        key_cols=key_cols,
                        engine_type=engine_type, execute=True, check=False)
            return df

        return _c

    return _auto_create


class Tushare(object):
    def __init__(self, Base_URL='http://api.waditu.com', token=None):
        self._base_url = Base_URL
        self.token = Token if token is None else token
        self._requests = partial(_get_req, token=Token)

    def __getattr__(self, item):
        return partial(_get_req, token=Token, api_name=item)

    def shortcut_stock_basic(self, **params):
        """
        stock_basic

        ts_code	str	Y	TS代码
        symbol	str	Y	股票代码
        name	str	Y	股票名称
        area	str	Y	地域
        industry	str	Y	所属行业
        fullname	str	N	股票全称
        enname	str	N	英文全称
        cnspell	str	N	拼音缩写
        market	str	Y	市场类型（主板/创业板/科创板/CDR）
        exchange	str	N	交易所代码
        curr_type	str	N	交易货币
        list_status	str	N	上市状态 L上市 D退市 P暂停上市
        list_date	str	Y	上市日期
        delist_date	str	N	退市日期
        is_hs	str	N	是否沪深港通标的，N否 H沪股通 S深股通
        """
        fields = 'ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_date,list_status,delist_date,is_hs'
        # params = dict()

        return self.stock_basic(fields, **params, )

    def shortcut_daily(self, **params):
        """

        输入参数

        名称	类型	必选	描述
        ts_code	str	N	股票代码（支持多个股票同时提取，逗号分隔）
        trade_date	str	N	交易日期（YYYYMMDD）
        start_date	str	N	开始日期(YYYYMMDD)
        end_date	str	N	结束日期(YYYYMMDD)

        输出参数

        名称	类型	描述
        ts_code	str	股票代码
        trade_date	str	交易日期
        open	float	开盘价
        high	float	最高价
        low	float	最低价
        close	float	收盘价
        pre_close	float	昨收价
        change	float	涨跌额
        pct_chg	float	涨跌幅 （未复权，如果是复权请用 通用行情接口 ）
        vol	float	成交量 （手）
        amount	float	成交额 （千元）
        """
        fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"

        return self.daily(fields, **params, )

    def shortcut_index_basic(self, **params):
        """
        输入参数

        名称	类型	必选	描述
        ts_code	str	N	指数代码
        name	str	N	指数简称
        market	str	N	交易所或服务商(默认SSE)
        publisher	str	N	发布商
        category	str	N	指数类别

        名称	类型	描述
        ts_code	str	TS代码
        name	str	简称
        fullname	str	指数全称
        market	str	市场
        publisher	str	发布方
        index_type	str	指数风格
        category	str	指数类别
        base_date	str	基期
        base_point	float	基点
        list_date	str	发布日期
        weight_rule	str	加权方式
        desc	str	描述
        exp_date	str	终止日期


        市场说明(market)

        市场代码	说明
        MSCI	MSCI指数
        CSI	中证指数
        SSE	上交所指数
        SZSE	深交所指数
        CICC	中金指数
        SW	申万指数
        OTH	其他指数


        指数列表

        主题指数
        规模指数
        策略指数
        风格指数
        综合指数
        成长指数
        价值指数
        有色指数
        化工指数
        能源指数
        其他指数
        外汇指数
        基金指数
        商品指数
        债券指数
        行业指数
        贵金属指数
        农副产品指数
        软商品指数
        油脂油料指数
        非金属建材指数
        煤焦钢矿指数
        谷物指数


        :param params:
        :return:


        ts_code	str	TS代码
        name	str	简称
        fullname	str	指数全称
        market	str	市场
        publisher	str	发布方
        index_type	str	指数风格
        category	str	指数类别
        base_date	str	基期
        base_point	float	基点
        list_date	str	发布日期
        weight_rule	str	加权方式
        desc	str	描述
        exp_date	str	终止日期
        """

        fields = "ts_code,name,fullname,market,publisher,index_type,category,base_date,base_point,list_date,weight_rule,desc,exp_date"

        return self.index_basic(fields, **params, )


def insert(df, table, db='tushare_data', key_cols=['ts_code'], engine_type='ReplacingMergeTree'):
    create(df, table, db=db, key_cols=key_cols, engine_type=engine_type)

    Node.insert_df(df, db, table, )


def create(df, table, db='tushare_data', key_cols=['ts_code'], engine_type='ReplacingMergeTree'):
    Node.create(db,
                table,
                df_or_sql_or_dict=df,
                key_cols=key_cols,
                engine_type=engine_type, execute=True, check=False)


def update_stock_quote():
    ts = Tushare()

    ts_code = ts.shortcut_stock_basic()['ts_code'].unique().tolist()
    ts_code2 = Node('select distinct ts_code from tushare_data.daily')['ts_code'].unique().tolist()
    tasks = set(ts_code) - set(ts_code2)

    for code in process_bar(tasks):
        print(code)
        df = ts.shortcut_daily(ts_code=code)
        insert(df, 'daily', db='tushare_data', key_cols=['ts_code', 'trade_date'])
        time.sleep(1)


if __name__ == '__main__':
    ts = Tushare()
    idx_info = ts.shortcut_index_basic()
    # insert(idx_info, 'index_basic', db='tushare_data', key_cols=['ts_code'], engine_type='ReplacingMergeTree')
    pass
