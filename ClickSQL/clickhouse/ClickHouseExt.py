# coding=utf-8
import pandas as pd
import re

from ClickSQL.clickhouse.ClickHouse import ClickHouseTableNode
from ClickSQL.errors import ClickHouseTableExistsError, ParameterTypeError


class CreateTableUtils(object):
    @staticmethod
    def _create_table_sql(db: str, table: str, dtypes_dict: dict,
                          order_by_key_cols: (list, tuple),
                          primary_key_cols=None, sample_expr=None,
                          engine_type: str = 'ReplacingMergeTree', partitions_expr=None,
                          settings="SETTINGS index_granularity = 8192", other=''):
        """

        :param db:
        :param table:
        :param dtypes_dict:
        :param order_by_key_cols:
        :param primary_key_cols:
        :param sample_expr:
        :param engine_type:
        :param partitions_expr:
        :param settings:
        :param other:
        :return:
        """
        """CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
                   (
                       name1 [type1] [DEFAULT|MATERIALIZED|ALIAS expr1],
                       name2 [type2] [DEFAULT|MATERIALIZED|ALIAS expr2],
                       ...
                   ) ENGINE = ReplacingMergeTree([ver])
                   [PARTITION BY expr]
                   [ORDER BY expr]
                   [PRIMARY KEY expr]
                   [SAMPLE BY expr]
                   [SETTINGS name=value, ...]"""

        if partitions_expr is None:
            partition_by_clause = ''
        else:
            partition_by_clause = f"PARTITION BY {partitions_expr}"

        if primary_key_cols is None:
            primary_by_clause = ''
        else:
            primary_key_expr = ','.join(primary_key_cols)
            primary_by_clause = f"PRIMARY BY ({primary_key_expr})"

        if sample_expr is None:
            sample_clause = ''
        else:
            sample_clause = sample_expr

        cols_def = ','.join([f"{name} {d_type}" for name, d_type in dtypes_dict.items()])

        main_body = f"CREATE TABLE IF NOT EXISTS {db}.{table} ( {cols_def} ) ENGINE = {engine_type}"

        ORDER_BY_CLAUSE = f"ORDER BY ( {','.join(order_by_key_cols)} )"

        cond_clause = f" {partition_by_clause} {ORDER_BY_CLAUSE} {primary_by_clause} {sample_clause} "

        base = f"{main_body} {cond_clause}  {other} {settings} "

        return base

    @staticmethod
    def _check_end_with_limit(string, pattern=r'[\s]+limit[\s]+[0-9]+$'):
        m = re.findall(pattern, string)
        if m is None or m == []:
            return False
        else:
            return True

    @staticmethod
    def translate_dtypes1_as_dtypes2(df: pd.DataFrame, src2target={'category': 'str'}):
        dtypes_series = df.dtypes
        for src, dest in src2target.items():
            if src in dtypes_series:
                category_cols = dtypes_series[dtypes_series == src].index
                for col in category_cols:
                    df[col] = df[col].astype(dest)
            else:
                pass
        return df

    @staticmethod
    def translate_dtypes_from_df(df: pd.DataFrame,
                                 translate_dtypes: dict = {'object': 'String', 'datetime64[ns]': 'Datetime'}):

        if hasattr(df, 'dtypes'):
            dtypes_series = df.dtypes.replace(translate_dtypes)
            return dtypes_series.map(lambda x: str(x).capitalize()).to_dict()
        elif hasattr(df, '_columns_') and 'type' in df._columns_ and 'name' in df._columns_:
            dtypes_series = df.set_index('name')['type'].replace(translate_dtypes)
            return dtypes_series.map(lambda x: str(x)).to_dict()
        else:
            raise ValueError(f'unknown df:{type(df)}')

    @classmethod
    def _create_table_from_df(cls, db: str, table: str, df: pd.DataFrame, key_cols: (list, tuple),
                              primary_key_cols=None, sample_expr=None,
                              engine_type: str = 'ReplacingMergeTree', extra_format_dict=None, partitions_expr=None,
                              settings="SETTINGS index_granularity = 8192",
                              other=''):

        df = cls.translate_dtypes1_as_dtypes2(df, src2target={'category': 'str'})
        cols = df.columns
        dtypes_dict = cls.translate_dtypes_from_df(df)
        if extra_format_dict is None:
            pass
        else:
            dtypes_dict.update(extra_format_dict)
        dtypes_dict = {k: v for k, v in dtypes_dict.items() if k in cols}
        base = cls._create_table_sql(db, table, dtypes_dict, key_cols,
                                     primary_key_cols=primary_key_cols,
                                     sample_expr=sample_expr,
                                     engine_type=engine_type,
                                     partitions_expr=partitions_expr,
                                     settings=settings,
                                     other=other
                                     )
        return base

    @classmethod
    def _obtain_describe_sql(cls, sql, pattern=r'[\s]+limit[\s]+[0-9]+$'):
        """
        detect limit end and obtain describe sql
        :param sql:
        :param pattern:
        :return:
        """
        limit_status = cls._check_end_with_limit(sql, pattern=pattern)
        if limit_status:
            describe_sql = f' describe({sql}) '
        else:
            describe_sql = f'describe ( {sql} limit 1)'

        return describe_sql


class ClickHouseTableNodeExt(ClickHouseTableNode, CreateTableUtils):
    def __init__(self, conn_str: (str, dict, None) = None, **kwarg):
        super(ClickHouseTableNodeExt, self).__init__(conn_str=conn_str, **kwarg)

    def create_view_if_not_exists(self, db: str, table: str, sql: str,
                                  key_cols: list,
                                  primary_key_cols=None, sample_expr=None,
                                  engine_type: str = 'view',
                                  extra_format_dict: (dict, None) = None, partitions_expr: (str, None) = None):
        if not self._check_exists(f'{db}.{table}', mode='table'):
            pass
        else:
            raise ClickHouseTableExistsError(f'{db}.{table} is exists!')
        if engine_type.lower() == 'view':
            create_sql = f'create view if not exists {db}.{table} as {sql}'
            self.query(create_sql)
        else:
            raise ValueError(f'current not support material view : {engine_type}')

    def create_table_if_not_exists(self, db: str, table: str, sql: (str, pd.DataFrame, None),
                                   key_cols: list,
                                   primary_key_cols=None, sample_expr=None,
                                   engine_type: str = 'ReplacingMergeTree',
                                   extra_format_dict: (dict, None) = None, partitions_expr: (str, None) = None):
        """

        :param db:
        :param table:
        :param sql:
        :param key_cols:
        :param primary_key_cols:
        :param sample_expr:
        :param engine_type:
        :param extra_format_dict:
        :param partitions_expr:
        :return:
        """
        if self._check_exists(f'{db}.{table}', mode='table'):
            pass
        else:
            self._create_table(db, table, sql, key_cols,
                               primary_key_cols=primary_key_cols,
                               sample_expr=sample_expr,
                               engine_type=engine_type,
                               extra_format_dict=extra_format_dict,
                               partitions_expr=partitions_expr)

    def _create_table(self, db: str, table: str, sql: (str, pd.DataFrame, None), key_cols: list,
                      primary_key_cols=None, sample_expr=None,
                      engine_type: str = 'ReplacingMergeTree',
                      extra_format_dict: (dict, None) = None, partitions_expr: (str, None) = None) -> object:

        # if sql is None:  # create table from sql
        #     create_sql = self._create_table_from_sql(db, table, None, key_cols,
        #                                              primary_key_cols=primary_key_cols, sample_expr=sample_expr,
        #                                              engine_type=engine_type,
        #                                              extra_format_dict=extra_format_dict,
        #                                              partitions_expr=partitions_expr,
        #                                              )
        if isinstance(sql, str):
            create_sql = self._create_table_from_sql(db, table, sql, key_cols,
                                                     primary_key_cols=primary_key_cols,
                                                     sample_expr=sample_expr,
                                                     engine_type=engine_type,
                                                     extra_format_dict=extra_format_dict,
                                                     partitions_expr=partitions_expr)
        elif isinstance(sql, pd.DataFrame):
            create_sql = self._create_table_from_df(db, table, sql, key_cols,
                                                    primary_key_cols=primary_key_cols,
                                                    sample_expr=sample_expr,
                                                    engine_type=engine_type,
                                                    extra_format_dict=extra_format_dict,
                                                    partitions_expr=partitions_expr)
        else:
            raise ParameterTypeError(f'unknown sql type: {type(sql)} @ {sql}')

        exist_status = self._check_exists(db_table=f'{db}.{table}')
        if exist_status:
            raise ClickHouseTableExistsError(f'{db}.{table} is exists!')
        self.query(create_sql)

    # def _create_table_from_df(self, db: str, table: str, df: pd.DataFrame, key_cols: (list, tuple),
    #                           primary_key_cols=None, sample_expr=None,
    #                           engine_type: str = 'ReplacingMergeTree', extra_format_dict=None, partitions_expr=None,
    #                           settings="SETTINGS index_granularity = 8192",
    #                           other=''):
    #
    #     df = self.translate_dtypes1_as_dtypes2(df, src2target={'category': 'str'})
    #     cols = df.columns
    #     dtypes_dict = self.translate_dtypes_from_df(df)
    #     if extra_format_dict is None:
    #         pass
    #     else:
    #         dtypes_dict.update(extra_format_dict)
    #     dtypes_dict = {k: v for k, v in dtypes_dict.items() if k in cols}
    #     base = self._create_table_sql(db, table, dtypes_dict, key_cols,
    #                                   primary_key_cols=primary_key_cols,
    #                                   sample_expr=sample_expr,
    #                                   engine_type=engine_type,
    #                                   partitions_expr=partitions_expr,
    #                                   settings=settings,
    #                                   other=other
    #                                   )
    # exist_status = self._check_exists(db_table=f'{db}.{table}')
    # if exist_status:
    #     raise ClickHouseTableExistsError(f'{db}.{table} is exists!')
    # # self.query(base)
    # return base

    def _create_table_from_sql(self, db: str, table: str, sql: str, key_cols: list,
                               extra_format_dict: (dict, None) = None,
                               primary_key_cols=None, sample_expr=None,
                               engine_type: str = 'ReplacingMergeTree',
                               settings="SETTINGS index_granularity = 8192",
                               other='',
                               partitions_expr=None):
        """


        :param db:
        :param table:
        :param sql:
        :param key_cols:
        :param engine_type:
        :param extra_format_dict:
        :return:
        """

        if isinstance(sql, str):
            # if self._check_sql_type(sql) == 'insert-liked':
            if sql.lower().startswith('create'):
                # if sql is create table sql will return directly
                return sql
        else:
            raise ParameterTypeError('sql must be string')
        # detect end with limit
        # limit_status = self._check_end_with_limit(sql, pattern=r'[\s]+limit[\s]+[0-9]+$')
        # if limit_status:
        #     describe_sql = f' describe({sql}) '
        # else:
        #     describe_sql = f'describe ( {sql} limit 1)'
        describe_sql = self._obtain_describe_sql(sql)
        dtypes_df = self.query(describe_sql)
        dtypes_dict = dict(dtypes_df[['name', 'type']].drop_duplicates().values)

        if isinstance(extra_format_dict, dict):
            dtypes_dict.update(extra_format_dict)
        else:
            pass
        sql = self._create_table_sql(db, table, dtypes_dict, key_cols,
                                     primary_key_cols=primary_key_cols,
                                     sample_expr=sample_expr,
                                     engine_type=engine_type,
                                     partitions_expr=partitions_expr,
                                     settings=settings,
                                     other=other)

        # self.query(sql)
        return sql


if __name__ == '__main__':
    pass
