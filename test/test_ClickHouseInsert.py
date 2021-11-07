# coding=utf-8
import unittest

import pandas as pd

from ClickSQL import ClickHouseTableNodeExt
from test.conn import conn


class MyTestCase(unittest.TestCase):
    @staticmethod
    def connection():
        fdg = ClickHouseTableNodeExt(conn_str=conn)

        # self.assertIsInstance(fdg, TableEngineCreator)
        return fdg

    def test_insert_df(self):
        data = {"date": {"0": "2016-12-01 00:00:00", "1": "2016-12-02 00:00:00", "2": "2016-12-03 00:00:00",
                         "3": "2016-12-04 00:00:00", "4": "2016-12-05 00:00:00"},
                "ADVISORS advisor fund__unscaled": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0},
                "ADVISORS advisor fund__monthly": {"0": 3.0, "1": 3.0, "2": 3.0, "3": 3.0, "4": 3.0},
                "scale": {"0": 0.481865285, "1": 0.481865285, "2": 0.481865285, "3": 0.481865285, "4": 0.481865285},
                "ADVISORS advisor fund_": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}}
        data = pd.DataFrame.from_dict(data)
        self.connection().create('test','test',data,key_cols=['date'],execute=True,check=False)
        self.connection().insert_df(data, 'test', 'test')
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
