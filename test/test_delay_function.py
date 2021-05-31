# coding=utf-8
import unittest

from ClickSQL.nodes.base import BaseSingleFactorTableNode
from test.conn import conn


class MyTestCaseDelayTasks(unittest.TestCase):
    def test_DelayTasks(self):
        FT = BaseSingleFactorTableNode(conn)
        FT('show tables', delay=True)

        self.assertEqual(len(FT.delay_tasks), 1)
        del FT

    def test_DelayTasks2(self):
        FT = BaseSingleFactorTableNode(conn)
        FT('show tables', delay=True)
        FT('show tables from system', delay=True)

        self.assertEqual(len(FT.delay_tasks), 2)
        del FT

    def test_DelayTasks3_delay_False(self):
        FT = BaseSingleFactorTableNode(conn)
        res = FT('show tables', delay=False)
        FT('show tables', delay=True)
        res2 = FT.delay_tasks.run(no_yield=True)[0]

        self.assertEqual(len(FT.delay_tasks), 1)
        self.assertEqual(all(res == res2), True)

    def test_DelayTasks4_delay_False(self):
        FT = BaseSingleFactorTableNode(conn)

        FT('show tables', delay=True)
        res = FT('show tables', delay=False)
        res2 = FT.delay_tasks.run(no_yield=True)[0]

        self.assertEqual(len(FT.delay_tasks), 1)
        self.assertEqual(all(res == res2), True)

    def test_delay_run(self):
        FT = BaseSingleFactorTableNode(conn)
        FT('show tables', delay=True)
        FT('show tables from system', delay=True)

        res = FT.delay_tasks.run(no_yield=True)

        self.assertEqual(len(res), 2)

    def test_delay_run_yield(self):
        FT = BaseSingleFactorTableNode(conn)
        FT('show tables', delay=True)
        FT('show tables from system', delay=True)

        res = FT.delay_tasks.run(no_yield=False)

        from collections import Iterable
        self.assertIsInstance(res, Iterable)

        # self.assertEqual(len(res), 2)

    def test_delay_run2(self):
        FT = BaseSingleFactorTableNode(conn)
        FT('show tables', delay=True)
        FT('show tables from system', delay=True)

        res = FT.delay_tasks.run(no_yield=True)

        self.assertEqual(True, all(res[0] == FT('show tables')))
        self.assertEqual(True, all(res[1] == FT('show tables from system')))


if __name__ == '__main__':
    unittest.main()
