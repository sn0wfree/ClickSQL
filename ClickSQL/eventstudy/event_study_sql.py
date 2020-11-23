# coding=utf-8
from ClickSQL.nodes.factor_node import BaseSingleFactorTableNode

es_data = BaseSingleFactorTableNode(
    'clickhouse://default:Imsn0wfree@47.104.186.157:8123/event_study.es',
    cols=['Date', 'Code', 'chg_rate'],
    order_by_cols=['Date asc'])

event_data = BaseSingleFactorTableNode(
    'clickhouse://default:Imsn0wfree@47.104.186.157:8123/event_study.event',
    cols=['CompanyName', 'Date'],
    order_by_cols=['Date asc'])

data = es_data.fetch_all()
event = event_data.fetch_all()
stock_list = event['CompanyName'].unique().tolist()

if __name__ == '__main__':

    pass
