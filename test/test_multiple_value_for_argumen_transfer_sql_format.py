# coding=utf-8
import unittest

from ClickSQL import BaseSingleFactorTableNode
from test.conn import conn

import pandas as pd


class MyTestCaseClickHouseExt(unittest.TestCase):
    def node(self):
        conn_str = conn
        return BaseSingleFactorTableNode(conn_str)

    def test_mp(self):
        test = pd.read_excel('SH000001-1m.xlsx')
        test['code'] = '000001.sh'
        test['date'] = pd.to_datetime(test['date'])
        print('load data')

        Node = self.node()

        Node.create('test',
                    'test1',
                    test,
                    key_cols=['code', 'date'], execute=True, check=False)

        Node.insert_df(test, 'test', 'test1')


if __name__ == '__main__':
    unittest.main()
