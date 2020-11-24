# coding=utf-8
import unittest

from ClickSQL import ClickHouseTableNodeExt
from test import conn


class MyTestCaseClickHouseExt(unittest.TestCase):
    @staticmethod
    def connection():
        fdg = ClickHouseTableNodeExt(conn_str=conn)

        # self.assertIsInstance(fdg, TableEngineCreator)
        return fdg

    def test_get_sql(self):
        node = self.connection()

        cols = []

        data_filter = {}

        db_table = 'raw.v_st_dis_buy_info'
        include_filter = True
        limit = 'limit 10'
        order_by_cols = ['money asc']

        other_filters = {'money': 'money >= 1000000'}

        sql = node.get_sql(db_table, cols=cols,
                           order_by_cols=order_by_cols,
                           data_filter=data_filter, include_filter=include_filter,
                           limit=limit,
                           **other_filters)

        real_real = 'select biz_channel_orderid,channel_no,core_date,cr_dr_type,cust_no,end_time,in_account,m_five_account_num,maturity_date,merchant_no,money,money_type,option_type,order_code,order_term,out_account,product_id,product_name,return_flow,s_account_num,send_flow,settle_account_num,start_time,trans_rate,trans_status,trans_type from raw.v_st_dis_buy_info where 1 and money >= 1000000  order by money asc limit 10 '
        self.assertEqual(real_real, sql)

        print(1)


if __name__ == '__main__':
    unittest.main()
