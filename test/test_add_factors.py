import unittest

from ClickSQL.factor_table.factortable import FactorTable
from test.conn import conn


class MyTestCaseAddFactors(unittest.TestCase):
    def test_100(self):
        ft = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')

        # ft.set_cik_dt([datetime.datetime.now()])
        res = ft.fetch(reduced=True, add_limit=True)
        self.assertEqual(res.shape[0], 100)

    def test_add_factor_from_db_table(self):
        ft = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')

        # ft.set_cik_dt([datetime.datetime.now()])
        res = ft.fetch(reduced=True, add_limit=True)
        self.assertEqual(res.shape[0], 100)

    def test_add_factor_from_sql(self):
        ft = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)
        ft.add_factor('select * from EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')

        # ft.set_cik_dt([datetime.datetime.now()])
        res = ft.fetch(reduced=True, add_limit=True)
        self.assertEqual(res.shape[0], 100)

    def test_add_factor_from_ft(self):
        ft = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')

        ft2 = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)
        ft2.add_factor('select * from EDGAR_LOG.edgar_log', factor_names=['norefer', 'size'], cik_dt='date',
                       cik_iid='ip')
        ft.add_factor(ft2, factor_names=['norefer'], cik_dt='cik_dt', cik_iid='cik_iid')

        res = ft.fetch(reduced=True, add_limit=True)
        self.assertEqual(res.shape[0], 100)

    def test_add_factor_from_df(self):
        ft = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')
        res = ft.fetch(reduced=True, add_limit=True)

        ft2 = FactorTable(conn, cik_dt='datetime', cik_iid='code', strict_cik=False)
        ft2.add_factor('EDGAR_LOG.edgar_log', factor_names=['idx', 'norefer'], cik_dt='date', cik_iid='ip')
        ft2.add_factor(res, factor_names=['size'], cik_dt='cik_dt', cik_iid='cik_iid')
        s = ft2.fetch(reduced=True, add_limit=True)
        self.assertTupleEqual(s.shape, (100, 3))
        self.assertListEqual(s.columns.tolist(), ['idx', 'norefer', 'size'])

    def test_add_factor_order1(self):
        ft = FactorTable(conn, strict_cik=False)
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['idx'], cik_dt='date', cik_iid='ip')
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['norefer'], cik_dt='date', cik_iid='ip')
        s = ft.fetch(reduced=True, add_limit=True)
        self.assertTupleEqual(s.shape, (100, 3))
        self.assertListEqual(s.columns.tolist(), ['size', 'idx', 'norefer', ])

    def test_add_factor_order2(self):
        ft = FactorTable(conn, strict_cik=False)
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['size', 'idx'], cik_dt='date', cik_iid='ip')
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['norefer'], cik_dt='date', cik_iid='ip')
        s = ft.fetch(reduced=True, add_limit=True)
        self.assertTupleEqual(s.shape, (100, 3))
        self.assertListEqual(s.columns.tolist(), ['size', 'idx', 'norefer', ])

    def test_add_factor_order3(self):
        ft = FactorTable(conn, strict_cik=False)
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['size'], cik_dt='date', cik_iid='ip')
        ft.add_factor('EDGAR_LOG.edgar_log', factor_names=['idx', 'norefer'], cik_dt='date', cik_iid='ip')
        s = ft.fetch(reduced=True, add_limit=True)
        self.assertTupleEqual(s.shape, (100, 3))
        self.assertListEqual(s.columns.tolist(), ['size', 'idx', 'norefer', ])


if __name__ == '__main__':
    unittest.main()
