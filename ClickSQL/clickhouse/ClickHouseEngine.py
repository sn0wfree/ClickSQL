# # coding=utf-8
# import pandas as pd
# from ClickSQL.utils.lazy_load import LazyInit
#
# from ClickSQL.clickhouse.ClickHouse import ClickHouseTableNode
#
#
# @LazyInit
# class MergeTree(object):
#     def __init__(self, db_table: str, dtypes_dict: dict, order_by_key_cols: (list, tuple), primary_key_cols=None,
#                  sample_expr=None, partitions_expr=None, settings="SETTINGS index_granularity = 8192"):
#         pass
#
#     @classmethod
#     def _create_table_sql(cls, db_table: str, dtypes_dict: dict,
#                           order_by_key_cols: (list, tuple),
#                           primary_key_cols=None, sample_expr=None,
#                           engine_type: str = 'MergeTree', partitions_expr=None,
#                           settings="SETTINGS index_granularity = 8192", other=''):
#         """
#
#         :param db:
#         :param table:
#         :param dtypes_dict:
#         :param order_by_key_cols:
#         :param primary_key_cols:
#         :param sample_expr:
#         :param engine_type:
#         :param partitions_expr:
#         :param settings:
#         :param other:
#         :return:
#         """
#         """CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
#                    (
#                        name1 [type1] [DEFAULT|MATERIALIZED|ALIAS expr1],
#                        name2 [type2] [DEFAULT|MATERIALIZED|ALIAS expr2],
#                        ...
#                    ) ENGINE = ReplacingMergeTree([ver])
#                    [PARTITION BY expr]
#                    [ORDER BY expr]
#                    [PRIMARY KEY expr]
#                    [SAMPLE BY expr]
#                    [SETTINGS name=value, ...]"""
#
#         cols_def_str = ''
#         order_by_key_str = ','.join(order_by_key_cols)
#         primary_key_cols_str = ''
#         sample = ''
#         main_body = f"CREATE TABLE IF NOT EXISTS {db_table} ({cols_def_str}) ENGINE={engine_type}"
#         main_settings = f"ORDER BY ({order_by_key_str}) PRIMARY KEY ({primary_key_cols_str}) SAMPLE BY {sample} {settings}"
#
#         cols_def = ','.join([f"{name} {d_type}" for name, d_type in dtypes_dict.items()])
#
#         maid_body = f"CREATE TABLE IF NOT EXISTS {db}.{table} ( {cols_def} ) ENGINE = {engine_type}"
#
#         ORDER_BY_CLAUSE = f"ORDER BY ( {','.join(order_by_key_cols)} )"
#
#         if partitions_expr is None:
#             PARTITION_by_CLAUSE = ''
#         else:
#             PARTITION_by_CLAUSE = f"PARTITION BY {partitions_expr}"
#
#         if primary_key_cols is None:
#             PRIMARY_BY_CLAUSE = ''
#         else:
#             primary_key_expr = ','.join(primary_key_cols)
#             PRIMARY_BY_CLAUSE = f" primary key by ({primary_key_expr})"
#
#         if sample_expr is None:
#             SAMPLE_CLAUSE = ''
#         else:
#             SAMPLE_CLAUSE = sample_expr
#
#         base = f"{maid_body} {PARTITION_by_CLAUSE} {ORDER_BY_CLAUSE} {PRIMARY_BY_CLAUSE} {SAMPLE_CLAUSE} {other} {settings}"
#
#         return base
#
#
# class ClickHouseTableNodeExt(ClickHouseTableNode):
#     @classmethod
#     def _create_table(cls, obj: object, db: str, table: str, sql: str, key_cols: list,
#                       engine_type: str = 'ReplacingMergeTree',
#                       extra_format_dict: (dict, None) = None, partitions_expr: (str, None) = None) -> object:
#         if isinstance(sql, str):
#             return cls._create_table_from_sql(obj, db, table, sql, key_cols,
#                                               engine_type=engine_type,
#                                               extra_format_dict=extra_format_dict, partitions_expr=partitions_expr)
#         elif isinstance(sql, pd.DataFrame):
#             return cls._create_table_from_df(obj, db, table, sql, key_cols,
#                                              engine_type='ReplacingMergeTree',
#                                              extra_format_dict=extra_format_dict, partitions_expr=partitions_expr)
#         else:
#             raise ValueError(f'unknown sql:{sql}')
#
#     @classmethod
#     def _create_table_from_df(cls, obj: object, db: str, table: str, df: pd.DataFrame, key_cols: (list, tuple),
#                               engine_type: str = 'ReplacingMergeTree', extra_format_dict=None, partitions_expr=None):
#         query_func = obj.query
#
#         df = cls.translate_dtypes1_as_dtypes2(df, src2target={'category': 'str'})
#         cols = df.columns
#         dtypes_dict = cls.translate_dtypes_from_df(df)
#         if extra_format_dict is None:
#             pass
#         else:
#             dtypes_dict.update(extra_format_dict)
#         dtypes_dict = {k: v for k, v in dtypes_dict.items() if k in cols}
#         base = cls._create_table_from_sql(db, table, dtypes_dict, key_cols, engine_type=engine_type,
#                                           extra_format_dict=extra_format_dict, partitions_expr=partitions_expr)
#         exist_status = cls._check_table_exists(obj, db, table)
#
#         query_func(base)
#         return exist_status
#
#     @classmethod
#     def _create_table_sql(cls, db: str, table: str, dtypes_dict: dict,
#                           order_by_key_cols: (list, tuple),
#                           primary_key_cols=None, sample_expr=None,
#                           engine_type: str = 'ReplacingMergeTree', partitions_expr=None,
#                           settings="SETTINGS index_granularity = 8192", other=''):
#         """
#
#         :param db:
#         :param table:
#         :param dtypes_dict:
#         :param order_by_key_cols:
#         :param primary_key_cols:
#         :param sample_expr:
#         :param engine_type:
#         :param partitions_expr:
#         :param settings:
#         :param other:
#         :return:
#         """
#         """CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
#                    (
#                        name1 [type1] [DEFAULT|MATERIALIZED|ALIAS expr1],
#                        name2 [type2] [DEFAULT|MATERIALIZED|ALIAS expr2],
#                        ...
#                    ) ENGINE = ReplacingMergeTree([ver])
#                    [PARTITION BY expr]
#                    [ORDER BY expr]
#                    [PRIMARY KEY expr]
#                    [SAMPLE BY expr]
#                    [SETTINGS name=value, ...]"""
#
#         cols_def = ','.join([f"{name} {d_type}" for name, d_type in dtypes_dict.items()])
#
#         maid_body = f"CREATE TABLE IF NOT EXISTS {db}.{table} ( {cols_def} ) ENGINE = {engine_type}"
#
#         ORDER_BY_CLAUSE = f"ORDER BY ( {','.join(order_by_key_cols)} )"
#
#         if partitions_expr is None:
#             PARTITION_by_CLAUSE = ''
#         else:
#             PARTITION_by_CLAUSE = f"PARTITION BY {partitions_expr}"
#
#         if primary_key_cols is None:
#             PRIMARY_BY_CLAUSE = ''
#         else:
#             primary_key_expr = ','.join(primary_key_cols)
#             PRIMARY_BY_CLAUSE = f"PARTITION BY ({primary_key_expr})"
#
#         if sample_expr is None:
#             SAMPLE_CLAUSE = ''
#         else:
#             SAMPLE_CLAUSE = sample_expr
#
#         base = f"{maid_body} {PARTITION_by_CLAUSE} {ORDER_BY_CLAUSE} {PRIMARY_BY_CLAUSE} {SAMPLE_CLAUSE} {other} {settings}"
#
#         return base
#
#     @staticmethod
#     def _check_end_with_limit(string, pattern=r'[\s]+limit[\s]+[0-9]+$'):
#         m = re.findall(pattern, string)
#         if m is None or m == []:
#             return False
#         else:
#             return True
#
#     @classmethod
#     def _create_table_from_sql(cls, db: str, table: str, sql: str, key_cols: list,
#                                extra_format_dict: (dict, None) = None,
#                                primary_key_cols=None, sample_expr=None, other='',
#                                engine_type: str = 'ReplacingMergeTree',
#                                partitions_expr=None, query_func=None):
#         """
#
#         :param obj:
#         :param db:
#         :param table:
#         :param sql:
#         :param key_cols:
#         :param engine_type:
#         :param extra_format_dict:
#         :return:
#         """
#
#         if isinstance(sql, str):
#             pass
#         else:
#             raise ValueError('sql must be string')
#
#         limit_status = cls._check_end_with_limit(sql, pattern=r'[\s]+limit[\s]+[0-9]+$')
#         if limit_status:
#             describe_sql = f' describe({sql}) '
#         else:
#             describe_sql = f'describe ( {sql} limit 1)'
#
#         if query_func is None:
#             raise ValueError('query function should be set!')
#
#         dtypes_df = query_func(describe_sql)
#
#         dtypes_dict = dict(dtypes_df[['name', 'type']].drop_duplicates().values)
#         if extra_format_dict is None:
#             pass
#         else:
#             dtypes_dict.update(extra_format_dict)
#         sql = cls._create_table_sql(db, table, dtypes_dict, key_cols, engine_type=engine_type,
#                                     primary_key_cols=primary_key_cols, sample_expr=sample_expr,
#                                     partitions_expr=partitions_expr, other=other)
#
#         return sql
#
#     @staticmethod
#     def translate_dtypes1_as_dtypes2(df: pd.DataFrame, src2target={'category': 'str'}):
#         dtypes_series = df.dtypes
#         for src, dest in src2target.items():
#             if src in dtypes_series:
#                 category_cols = dtypes_series[dtypes_series == src].index
#                 for col in category_cols:
#                     df[col] = df[col].astype(dest)
#             else:
#                 pass
#         return df
#
#     @staticmethod
#     def translate_dtypes_from_df(df: pd.DataFrame, translate_dtypes: dict = {'object': 'String',
#                                                                              'datetime64[ns]': 'Datetime'}):
#         if hasattr(df, 'dtypes'):
#             dtypes_series = df.dtypes.replace(translate_dtypes)
#             return dtypes_series.map(lambda x: str(x).capitalize()).to_dict()
#         elif hasattr(df, '_columns_') and 'type' in df._columns_ and 'name' in df._columns_:
#             dtypes_series = df.set_index('name')['type'].replace(translate_dtypes)
#             return dtypes_series.map(lambda x: str(x)).to_dict()
#         else:
#             raise ValueError(f'unknown df:{type(df)}')
#
#     @classmethod
#     def _create_table_from_df(cls, db: str, table: str, df: pd.DataFrame, key_cols: (list, tuple),
#                               engine_type: str = 'ReplacingMergeTree', extra_format_dict=None, partitions_expr=None,
#                               src2target={'category': 'str'},
#                               query_func=None
#                               ):
#
#         df = cls.translate_dtypes1_as_dtypes2(df, src2target={'category': 'str'})
#         cols = df.columns
#         dtypes_dict = cls.translate_dtypes_from_df(df)
#         if extra_format_dict is None:
#             pass
#         else:
#             dtypes_dict.update(extra_format_dict)
#         dtypes_dict = {k: v for k, v in dtypes_dict.items() if k in cols}
#         base = cls._create_table_from_sql(db, table, dtypes_dict, key_cols, engine_type=engine_type,
#                                           extra_format_dict=extra_format_dict, partitions_expr=partitions_expr)
#         exist_status = cls._check_table_exists(obj, db, table)
#
#         query_func(base)
#         return exist_status
#
#     @classmethod
#     def _check_table_exists(cls, obj, db, table):
#         ## todo check the table exists
#         pass
