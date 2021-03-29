# coding=utf-8

from ClickSQL.factor_table.factortable import FatctorTable

conn = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/test.test'
Node = FatctorTable(conn, cik_dt='datetime', cik_iid='code')

if __name__ == '__main__':
    Node.add_factor('test.test', factor_names='test1', )
    Node.add_factor('test.test', factor_names='test2', )
    f = Node.__auto_merge__()
    print(f)
    pass
