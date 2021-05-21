# coding=utf-8

from ClickSQL.factor_table.factortable import FactorTable

from test.conn import conn

Node = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)

if __name__ == '__main__':
    Node.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')
    # Node.add_factor('EDGAR_LOG.edgar_log', factor_names='cik', cik_dt='date', cik_iid='ip')
    # Node.add_factor('test.test2', factor_names='test2', cik_dt='cik_dt', cik_iid='cik_iid')
    # f = Node.fetch(reduced=True, add_limit=True)
    res = Node.fetch(reduced=True, add_limit=True)
    print(res)
    pass
