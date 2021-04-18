import unittest

from ClickSQL.nodes.base import BaseSingleFactorTableNode
from test.conn import conn


class MyTestCaseFactorTable(unittest.TestCase):
    def test_add_factor_2way(self):
        FT = BaseSingleFactorTableNode(conn, cik_dt='DATE', cik_iid='rank')
        FT2 = BaseSingleFactorTableNode(conn, cik_dt='DATE', cik_iid='rank')

        FT._add_factors_ftmod_system('q_factors')
        FT2._add_factors_ftmod_map('q_factors')
        c1 = FT.show_factors()
        c2 = FT2.show_factors()
        self.assertEqual(True, all(c1 == c2))


if __name__ == '__main__':
    unittest.main()
