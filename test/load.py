# coding=utf-8

from ClickSQL.factor_table.factortable import FatctorTable

conn = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/test.test'
Node = FatctorTable(conn, cik_dt='datetime', cik_iid='code')

if __name__ == '__main__':
    Node.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')
    Node.add_factor('EDGAR_LOG.edgar_log', factor_names='cik', cik_dt='date', cik_iid='ip')
    # Node.add_factor('test.test2', factor_names='test2', cik_dt='cik_dt', cik_iid='cik_iid')
    f = Node.obtain_merged(reduced=False, add_limit=True)
    print(f)
    pass
