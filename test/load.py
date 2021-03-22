# coding=utf-8

from ClickSQL import BaseSingleFactorTableNode

conn = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/test.test'
Node = BaseSingleFactorTableNode(conn)

if __name__ == '__main__':
    Node.create()
    pass
