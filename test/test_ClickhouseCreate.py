# coding=utf-8
import unittest
import pandas as pd
import numpy as np
from ClickSQL.clickhouse.ClickHouseCreate import TableEngineCreator, ClickHouseTableExistsError

conn = 'clickhouse://default:12345@xxx.xxx.xxx.xxx:8123/test.test'


class MyTestCase(unittest.TestCase):
    @staticmethod
    def connection():
        fdg = TableEngineCreator(conn_str=conn)

        # self.assertIsInstance(fdg, TableEngineCreator)
        return fdg

    def test_select_create_existed(self):
        fg = self.connection()
        with self.assertRaises(ClickHouseTableExistsError) as f:
            out = fg.create(db='test',
                            table='test3',
                            df_or_sql_or_dict='select * from test.test4 limit 1',
                            key_cols=['test1'],
                            engine_type='ReplacingMergeTree',
                            primary_key_cols=None,
                            sample_by_cols=None,
                            extra_format_dict=None,
                            partition_by_cols=None, execute=False)
            # print(out)
            # self.assertEqual(out, '1')

    def test_select_create_sql(self):
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        df_or_sql_or_dict='select * from test.test4 limit 1',
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=None,
                        sample_by_cols=None,
                        extra_format_dict=None, check=False,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5   ENGINE = ReplacingMergeTree    ORDER BY  test1         SETTINGS index_granularity = 8192 as select * from test.test4 limit 1 '
        # print(out)
        self.assertEquals(out, res)

    def test_select_create_df(self):
        df = pd.DataFrame(np.random.random(size=(100, 2)), columns=['test1', 'test2'])
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        df_or_sql_or_dict=df,
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=None,
                        sample_by_cols=None,
                        extra_format_dict=None, check=False,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 Float64,test2 Float64 ) ENGINE = ReplacingMergeTree   ORDER BY  test1         SETTINGS index_granularity = 8192  '
        # print(out)
        self.assertEquals(out, res)

    def test_select_create_dtypes(self):
        # df = pd.DataFrame(np.random.random(size=(100, 2)), columns=['test1', 'test2'])
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        df_or_sql_or_dict={'test1': 'Float64', 'test2': 'Float64'},
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=None,
                        sample_by_cols=None,
                        extra_format_dict=None, check=False,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 Float64,test2 Float64 ) ENGINE = ReplacingMergeTree  ORDER BY  test1       SETTINGS index_granularity = 8192'
        # print(out)
        self.assertEquals(out, res)

    def test_select_create_sql_primary(self):
        # df = pd.DataFrame(np.random.random(size=(100, 2)), columns=['test1', 'test2'])
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        df_or_sql_or_dict={'test1': 'Float64', 'test2': 'Float64'},
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=['test1'],
                        sample_by_cols=None,
                        extra_format_dict=None, check=False,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 Float64,test2 Float64 ) ENGINE = ReplacingMergeTree  ORDER BY  test1   PRIMARY KEY  test1      SETTINGS index_granularity = 8192'
        print(out)
        self.assertEquals(out, res)

    def test_select_create_df_primary(self):
        df = pd.DataFrame(np.random.random(size=(100, 2)), columns=['test1', 'test2'])
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        df_or_sql_or_dict=df,
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=['test1'],
                        sample_by_cols=None,
                        extra_format_dict=None, check=False,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 Float64,test2 Float64 ) ENGINE = ReplacingMergeTree   ORDER BY  test1   PRIMARY KEY   test1        SETTINGS index_granularity = 8192  '
        # print(out)
        self.assertEquals(out, res)

    def test_select_create_dtypes_primary(self):
        # df = pd.DataFrame(np.random.random(size=(100, 2)), columns=['test1', 'test2'])
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        df_or_sql_or_dict={'test1': 'Float64', 'test2': 'Float64'},
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=['test1'],
                        sample_by_cols=None,
                        extra_format_dict=None, check=False,
                        partition_by_cols=None, execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 Float64,test2 Float64 ) ENGINE = ReplacingMergeTree  ORDER BY  test1   PRIMARY KEY  test1      SETTINGS index_granularity = 8192'
        print(out)
        self.assertEquals(out, res)

    def test_select_create_dtypes_primary_partition(self):
        # df = pd.DataFrame(np.random.random(size=(100, 2)), columns=['test1', 'test2'])
        fg = self.connection()
        out = fg.create(db='test',
                        table='test5',
                        df_or_sql_or_dict={'test1': 'Float64', 'test2': 'Float64'},
                        key_cols=['test1'],
                        engine_type='ReplacingMergeTree',
                        primary_key_cols=['test1'],
                        sample_by_cols=None,
                        extra_format_dict=None, check=False,
                        partition_by_cols=['toString(test1)'], execute=False)

        res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 Float64,test2 Float64 ) ENGINE = ReplacingMergeTree PARTITION BY  toString(test1)   ORDER BY  test1   PRIMARY KEY  test1      SETTINGS index_granularity = 8192'
        print(out)
        self.assertEquals(out, res)

    # def test_select_create(self):
    #     fg = self.connection()
    #     out = fg.create(db='test',
    #                     table='test5',
    #                     sql='select * from test.test4 limit 1',
    #                     key_cols=['test1'],
    #                     engine_type='ReplacingMergeTree',
    #                     primary_key_cols=['test1'],
    #                     sample_by_cols=None,
    #                     extra_format_dict=None, check=False,
    #                     partition_by_cols=None, execute=False)
    #
    #     res = 'CREATE TABLE IF NOT EXISTS test.test5  ( test1 String ) ENGINE = ReplacingMergeTree   ORDER BY ( test1 )  PRIMARY KEY ( test1 )  SETTINGS index_granularity = 8192 '
    #     # print(out)
    #     self.assertEquals(out, res)


if __name__ == '__main__':
    unittest.main()
