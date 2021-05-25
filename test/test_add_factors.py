import unittest
from ClickSQL.factor_table.factortable import FactorTable

from test.conn import conn
import datetime


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
        ft2.add_factor('select * from EDGAR_LOG.edgar_log', factor_names=['norefer','size'], cik_dt='date', cik_iid='ip')
        ft.add_factor(ft2, factor_names=['norefer'], cik_dt='cik_dt', cik_iid='cik_iid')

        res = ft.fetch(reduced=True, add_limit=True)
        self.assertEqual(res.shape[0], 100)


if __name__ == '__main__':
    unittest.main()
