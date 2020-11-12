import unittest

from ClickSQL.clickhouse.ClickHouseCreate import TableEngineCreator, ClickHouseTableExistsError


class MyTestCase(unittest.TestCase):
    def connection(self):
        conn = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/test.test'
        fdg = TableEngineCreator(conn_str=conn)

        # self.assertIsInstance(fdg, TableEngineCreator)
        return fdg

    def test_select_create_existed(self):
        fg = self.connection()
        with self.assertRaises(ClickHouseTableExistsError) as f:
            out = fg.create(db='test',
                            table='test3',
                            sql='select * from test.test4 limit 1',
                            key_cols=['test1'],
                            engine_type='ReplacingMergeTree',
                            primary_key_cols=None,
                            sample_by_cols=None,
                            extra_format_dict=None,
                            partition_by_cols=None, execute=False)
            # print(out)
            # self.assertEqual(out, '1')

    def test_select_create(self):
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        sql='select * from test.test4 limit 1',
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=None,
                        sample_by_cols=None,
                        extra_format_dict=None,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 String ) ENGINE = ReplacingMergeTree   ORDER BY ( test1 )        SETTINGS index_granularity = 8192 '
        # print(out)
        self.assertEquals(out, res)

    def test_select_create(self):
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        sql='select * from test.test4 limit 1',
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=['test1'],
                        sample_by_cols=None,
                        extra_format_dict=None,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 String ) ENGINE = ReplacingMergeTree   ORDER BY ( test1 )  PRIMARY KEY ( test1 )  SETTINGS index_granularity = 8192 '
        # print(out)
        self.assertEquals(out, res)


if __name__ == '__main__':
    unittest.main()
