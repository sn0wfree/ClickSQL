# coding=utf-8
import pandas as pd
import re

from ClickSQL.clickhouse.ClickHouse import ClickHouseTableNode
from ClickSQL.errors import ClickHouseTableExistsError, ParameterTypeError


class SQLBuilder(object):

    @staticmethod
    def _assemble_sample(sample=None):
        sample_clause = '' if sample is None else f'SAMPLE {sample}'
        return sample_clause

    @staticmethod
    def _assemble_array_join(array_join_list=None):

        array_join_clause = '' if array_join_list is None else f"ARRAY JOIN {', '.join(array_join_list)}"

        return array_join_clause

    @staticmethod
    def _assemble_join(join_info_dict=None):
        if join_info_dict is None:
            join_clause = ''
        else:
            join_type = join_info_dict.get('type')
            on_ = join_info_dict.get('ON')
            using_ = join_info_dict.get('USING')

            if join_type is None:
                raise ValueError('join_info_dict cannot locate join_type condition')

            if on_ is None:
                if using_ is None:
                    raise ValueError('join_info_dict cannot locate ON or USING condition')
                else:
                    join_clause = f'{join_type} USING ({using_})'
            else:
                join_clause = f'{join_type} ON {on_}'
        return join_clause

    @staticmethod
    def _assemble_where_like(a_list, prefix='WHERE'):

        sample_clause = '' if a_list is None else f"{prefix} {' and '.join(a_list)}"

        return sample_clause

    @staticmethod
    def _assemble_group_by(group_by_cols=None):
        if group_by_cols is None:
            sample_clause = ''
        else:
            group_by_cols_str = ','.join(group_by_cols)
            sample_clause = f'GROUP BY ({group_by_cols_str})'
        return sample_clause

    @staticmethod
    def _assemble_order_by(order_by_cols=None):

        sample_clause = '' if order_by_cols is None else f"ORDER BY ({','.join(order_by_cols)})"

        return sample_clause

    @staticmethod
    def _assemble_limit_by(limit_n_by_dict=None):

        if limit_n_by_dict is None:
            sample_clause = ''
        else:
            N = limit_n_by_dict['N']
            order_by_cols_str = ','.join(limit_n_by_dict['limit_by_cols'])
            sample_clause = f'LIMIT {N} BY ({order_by_cols_str})'
        return sample_clause

    @staticmethod
    def _assemble_limit(limit_n=None):

        if limit_n is None:
            SAMPLE_CLAUSE = ''
        else:
            SAMPLE_CLAUSE = f'LIMIT {limit_n} '
        return SAMPLE_CLAUSE

    @staticmethod
    def raw_create_select_sql(SELECT_CLAUSE: str, DB_TABLE: str, SAMPLE_CLAUSE: str, ARRAY_JOIN_CLAUSE: str,
                              JOIN_CLAUSE: str, PREWHERE_CLAUSE: str, WHERE_CLAUSE: str, GROUP_BY_CLAUSE: str,
                              HAVING_CLAUSE: str, ORDER_BY_CLAUSE: str, LIMIT_N_CLAUSE: str, LIMIT_CLAUSE: str):
        """

        :param SELECT_CLAUSE:
        :param DB_TABLE:
        :param SAMPLE_CLAUSE:
        :param ARRAY_JOIN_CLAUSE:
        :param JOIN_CLAUSE:
        :param PREWHERE_CLAUSE:
        :param WHERE_CLAUSE:
        :param GROUP_BY_CLAUSE:
        :param HAVING_CLAUSE:
        :param ORDER_BY_CLAUSE:
        :param LIMIT_N_CLAUSE:
        :param LIMIT_CLAUSE:
        :return:
        """
        """SELECT [DISTINCT] expr_list
                    [FROM [db.]table | (subquery) | table_function] [FINAL]
                    [SAMPLE sample_coeff]
                    [ARRAY JOIN ...]
                    [GLOBAL] ANY|ALL INNER|LEFT JOIN (subquery)|table USING columns_list
                    [PREWHERE expr]
                    [WHERE expr]
                    [GROUP BY expr_list] [WITH TOTALS]
                    [HAVING expr]
                    [ORDER BY expr_list]
                    [LIMIT n BY columns]
                    [LIMIT [n, ]m]
                    [UNION ALL ...]
                    [INTO OUTFILE filename]
                    [FORMAT format]"""
        if DB_TABLE.lower().startswith('select '):
            DB_TABLE = f"( {DB_TABLE} )"
        else:
            pass
        main_body = f"SELECT {SELECT_CLAUSE} FROM {DB_TABLE} {SAMPLE_CLAUSE}"
        join = f"{ARRAY_JOIN_CLAUSE} {JOIN_CLAUSE}"
        where_conditions = f"{PREWHERE_CLAUSE} {WHERE_CLAUSE} {GROUP_BY_CLAUSE} {HAVING_CLAUSE} "
        order_limit = f"{ORDER_BY_CLAUSE} {LIMIT_N_CLAUSE} {LIMIT_CLAUSE}"
        sql = f"{main_body} {join} {where_conditions} {order_limit}"
        return sql

    @classmethod
    def selector(cls, db_table: str, cols: list,
                 sample: (int, float, None) = None,
                 array_join: (list, None) = None,
                 join: (dict, None) = None,
                 prewhere: (list, None) = None,
                 where: (list, None) = None,
                 having: (list, None) = None,
                 group_by: (list, None) = None,
                 order_by: (list, None) = None,
                 limit_by: (dict, None) = None,
                 limit: (int, None) = None) -> str:
        """

        :param having: str ["r1 >1 and r2 <2"]
        :param db_table: str default.test
        :param cols: list [ r1,r2,r3 ]
        :param sample: str 0.1 or 1000
        :param array_join: list ['arrayA as a','arrayB as b']
        :param join: dict {'type':'all left join','USING' : "r1,r2"}
        :param prewhere: str ["r1 >1" , "r2 <2"]
        :param where: str ["r1 >1.5" , "r2 <1.3"]
        :param group_by: list ['r1','r2']
        :param order_by: list ['r1 desc','r2 desc']
        :param limit_by: dict {'N':10,'limit_by_cols':['r1','r2']}
        :param limit: int 100
        :return:  str
        """

        SELECT_CLAUSE = ','.join(cols)
        SAMPLE_CLAUSE = cls._assemble_sample(sample=sample)
        ARRAY_JOIN_CLAUSE = cls._assemble_array_join(array_join_list=array_join)
        JOIN_CLAUSE = cls._assemble_join(join)
        PREWHERE_CLAUSE = cls._assemble_where_like(prewhere, prefix='PREWHERE')
        WHERE_CLAUSE = cls._assemble_where_like(where, prefix='WHERE')
        HAVING_CLAUSE = cls._assemble_where_like(having, prefix='HAVING')
        GROUP_BY_CLAUSE = cls._assemble_group_by(group_by)
        ORDER_BY_CLAUSE = cls._assemble_order_by(order_by)
        LIMIT_N_CLAUSE = cls._assemble_limit_by(limit_by)
        LIMIT_CLAUSE = cls._assemble_limit(limit)

        return cls.raw_create_select_sql(SELECT_CLAUSE, db_table, SAMPLE_CLAUSE, ARRAY_JOIN_CLAUSE, JOIN_CLAUSE,
                                         PREWHERE_CLAUSE, WHERE_CLAUSE, GROUP_BY_CLAUSE, HAVING_CLAUSE, ORDER_BY_CLAUSE,
                                         LIMIT_N_CLAUSE, LIMIT_CLAUSE)


class CreateBuilder(object):
    @staticmethod
    def _assemble_conditions_clause(prefix_clause: str, cols: (list, tuple, None), default=''):
        """

        :param prefix_clause:
        :param cols:
        :param default:
        :return:
        """
        if cols is None:
            return default
        else:
            cols_str = ','.join(cols)
            if len(cols) >= 2:
                return f"{prefix_clause} ( {cols_str} ) "
            else:
                return f"{prefix_clause}  {cols_str}  "

    pass


class CreateTableFromSQLUtils(object):
    @classmethod
    def _obtain_describe_sql(cls, sql: str, pattern: str = r'[\s]+limit[\s]+[0-9]+$'):
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

    @staticmethod
    def _create_table_sql(db: str, table: str, select_or_dtypes_dict: (str, dict),
                          order_by_key_cols: (list, tuple),
                          primary_key_cols: (list, tuple, None) = None,
                          sample_by_cols: (list, tuple, None) = None,
                          partition_by_cols: (list, tuple, None) = None,
                          engine_type: str = 'ReplacingMergeTree',
                          on_cluster: str = '',
                          settings: str = "SETTINGS index_granularity = 8192",
                          other: str = '') -> str:
        """
        CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
                   (
                       name1 [type1] [DEFAULT|MATERIALIZED|ALIAS expr1],
                       name2 [type2] [DEFAULT|MATERIALIZED|ALIAS expr2],
                       ...
                   ) ENGINE = ReplacingMergeTree([ver])
                   [PARTITION BY expr]
                   [ORDER BY expr]
                   [PRIMARY KEY expr]
                   [SAMPLE BY expr]
                   [SETTINGS name=value, ...]

        :param db:
        :param table:
        :param select_or_dtypes_dict:
        :param order_by_key_cols:
        :param primary_key_cols:
        :param sample_expr:
        :param engine_type:
        :param partitions_expr:
        :param settings:
        :param other:
        :return:
        """

        default = ""

        # order_by_clause = f"ORDER BY ( {','.join(order_by_key_cols)} )"
        order_by_clause = CreateBuilder._assemble_conditions_clause('ORDER BY', order_by_key_cols, default=default)
        sample_clause = CreateBuilder._assemble_conditions_clause('SAMPLE BY', sample_by_cols, default=default)
        primary_by_clause = CreateBuilder._assemble_conditions_clause('PRIMARY KEY ', primary_key_cols, default=default)
        partition_by_clause = CreateBuilder._assemble_conditions_clause('PARTITION BY', partition_by_cols,
                                                                        default=default)

        # if partitions_expr is None:
        #     partition_by_clause = ''
        # else:
        #     partition_by_clause = f"PARTITION BY {partitions_expr}"

        # if sample_expr is None:
        #     sample_clause = ''
        # else:
        #     sample_clause = sample_expr

        # if primary_key_cols is None:
        #     primary_by_clause = ''
        # else:
        #     primary_key_expr = ','.join(primary_key_cols)
        #     primary_by_clause = f"PRIMARY BY ({primary_key_expr})"
        if isinstance(select_or_dtypes_dict, dict):
            cols_def = ','.join([f"{name} {d_type}" for name, d_type in select_or_dtypes_dict.items()])
            main_body = f"CREATE TABLE IF NOT EXISTS {db}.{table} {on_cluster} ( {cols_def} ) ENGINE = {engine_type}"
            tail = ''
        elif isinstance(select_or_dtypes_dict, str) and select_or_dtypes_dict.lower().startswith('select'):
            main_body = f"CREATE TABLE IF NOT EXISTS {db}.{table} {on_cluster}  ENGINE = {engine_type} "
            tail = f'as {select_or_dtypes_dict}'
        else:
            raise ParameterTypeError(
                f'select_or_dtypes_dict only accept dict or str start with select! but get {select_or_dtypes_dict}')

        cond_clause = f" {partition_by_clause} {order_by_clause} {primary_by_clause} {sample_clause} "

        base = f"{main_body} {cond_clause}  {other} {settings} {tail} "

        return base

    @staticmethod
    def _check_end_with_limit(string: str, pattern: str = r'[\s]+limit[\s]+[0-9]+$'):
        m = re.findall(pattern, string)
        if m is None or m == []:
            return False
        else:
            return True

    @staticmethod
    def _translate_dtypes1_as_dtypes2(df: pd.DataFrame, src2target: dict = {'category': 'str'}):
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
    def _translate_dtypes_from_df(df: pd.DataFrame,
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
                              primary_key_cols: (list, tuple, None) = None,
                              sample_by_cols: (list, tuple, None) = None,
                              partition_by_cols: (list, tuple, None) = None,
                              on_cluster: str = '',
                              engine_type: str = 'ReplacingMergeTree',
                              extra_format_dict: dict = None,
                              settings: str = "SETTINGS index_granularity = 8192",
                              other: str = ''):

        df = cls._translate_dtypes1_as_dtypes2(df, src2target={'category': 'str'})
        cols = df.columns
        dtypes_dict = cls._translate_dtypes_from_df(df)
        if extra_format_dict is None:
            pass
        else:
            dtypes_dict.update(extra_format_dict)
        dtypes_dict = {k: v for k, v in dtypes_dict.items() if k in cols}
        base = cls._create_table_sql(db, table, dtypes_dict, key_cols,
                                     primary_key_cols=primary_key_cols,
                                     sample_by_cols=sample_by_cols,
                                     engine_type=engine_type,
                                     on_cluster=on_cluster,
                                     partition_by_cols=partition_by_cols,
                                     settings=settings,
                                     other=other
                                     )
        return base

    @classmethod
    def _create_table_from_sql(cls, db: str, table: str, sql: str, key_cols: list,
                               extra_format_dict: (dict, None) = None,
                               primary_key_cols: (list, tuple, None) = None,
                               sample_by_cols: (list, tuple, None) = None,
                               partition_by_cols: (list, tuple, None) = None,
                               engine_type: str = 'ReplacingMergeTree',
                               on_cluster: str = '',
                               settings: str = "SETTINGS index_granularity = 8192",
                               other='',
                               ):
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
            elif sql.lower().startswith('select'):
                pass
            else:
                raise ParameterTypeError(f'sql must be string and start with select! but get {sql}')
        else:
            raise ParameterTypeError(f'sql must be string and start with select! but get {sql}')
        # detect end with limit
        # describe_sql = cls._obtain_describe_sql(sql)
        # dtypes_df = query(describe_sql)
        # dtypes_dict = dict(dtypes_df[['name', 'type']].drop_duplicates().values)

        # if isinstance(extra_format_dict, dict):
        #     dtypes_dict.update(extra_format_dict)
        # else:
        #     pass
        sql = cls._create_table_sql(db, table, sql, key_cols,
                                    primary_key_cols=primary_key_cols,
                                    sample_by_cols=sample_by_cols, on_cluster=on_cluster,
                                    engine_type=engine_type,
                                    partition_by_cols=partition_by_cols,
                                    settings=settings,
                                    other=other)

        # self.query(sql)
        return sql

    @classmethod
    def creator_sql(cls, db: str, table: str, df_or_sql: (pd.DataFrame, str), key_cols: (list, tuple),
                    extra_format_dict: (dict, None) = None,
                    primary_key_cols: (list, tuple, None) = None,
                    sample_by_cols: (list, tuple, None) = None,
                    partition_by_cols: (list, tuple, None) = None,
                    on_cluster='',
                    engine_type: str = 'ReplacingMergeTree',
                    settings: str = "SETTINGS index_granularity = 8192",
                    other: str = '',
                    ):
        if isinstance(df_or_sql, pd.DataFrame):
            return cls._create_table_from_df(db, table, df_or_sql, key_cols,
                                             primary_key_cols=primary_key_cols, sample_by_cols=sample_by_cols,
                                             engine_type=engine_type, extra_format_dict=extra_format_dict,
                                             partition_by_cols=partition_by_cols, on_cluster=on_cluster,
                                             settings=settings,
                                             other=other)
        else:
            return cls._create_table_from_sql(db, table, df_or_sql, key_cols, on_cluster=on_cluster,
                                              primary_key_cols=primary_key_cols, sample_by_cols=sample_by_cols,
                                              engine_type=engine_type, extra_format_dict=extra_format_dict,
                                              partition_by_cols=partition_by_cols,
                                              settings=settings,
                                              other=other)
        pass


class CreateTableFromInfoUtils(object):

    @staticmethod
    def _raw_create_table_sql(db_table: str,
                              cols_def: str,
                              order_by_clause: str,
                              primary_by_clause: str = '',
                              sample_clause: str = '',
                              engine_type: str = 'ReplacingMergeTree',
                              on_cluster: str = '',
                              partition_by_clause: str = '',
                              ttl: str = '',
                              settings="SETTINGS index_granularity = 8192"
                              ):
        # TODO add ttl expr at future
        """
        CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
                    (
                        name1 [type1] [DEFAULT|MATERIALIZED|ALIAS expr1],
                        name2 [type2] [DEFAULT|MATERIALIZED|ALIAS expr2],
                        ...
                    ) ENGINE = ReplacingMergeTree([ver])
                    [PARTITION BY expr]
                    [ORDER BY expr]
                    [PRIMARY KEY expr]
                    [SAMPLE BY expr]
                    [SETTINGS name=value, ...]
        :param ttl:
        :param on_cluster:
        :param sample_clause:
        :param primary_by_clause:
        :param partition_by_clause:
        :param db_table:
        :param cols_def:
        :param order_by_clause:
        :param engine_type:
        :return:
        """

        maid_body = f"CREATE TABLE IF NOT EXISTS {db_table} {on_cluster} ( {cols_def} ) ENGINE = {engine_type}"

        conds = f"{partition_by_clause} {order_by_clause} {primary_by_clause} {sample_clause} {ttl}"

        base = f"{maid_body} {conds}  {settings}"
        return base

    @classmethod
    def creator_info(cls,
                     db_table: str,
                     var_dict: dict,
                     order_by_cols: (list, tuple),
                     sample_by_cols: (list, tuple, None) = None,
                     partition_by_cols: (list, tuple, None) = None,
                     primary_by_cols: (list, tuple, None) = None,
                     on_cluster: str = '',
                     engine_type: str = 'ReplacingMergeTree',
                     ttl: str = '',
                     settings="SETTINGS index_granularity = 8192"
                     ):
        """

        :param settings:
        :param engine_type:
        :param var_dict:
        :param ttl:
        :param db_table:
        :param order_by_cols:
        :param sample_by_cols:
        :param on_cluster:
        :param partition_by_cols:
        :param primary_by_cols:
        :return:
        """
        default = ''
        cols_def = ','.join([f"{var} {v}" for var, v in var_dict.items()])

        order_by_clause = CreateBuilder._assemble_conditions_clause('ORDER BY', order_by_cols, default=default)
        sample_clause = CreateBuilder._assemble_conditions_clause('SAMPLE BY', sample_by_cols, default=default)
        primary_by_clause = CreateBuilder._assemble_conditions_clause('PRIMARY BY', primary_by_cols, default=default)
        partition_by_clause = CreateBuilder._assemble_conditions_clause('PARTITION BY', partition_by_cols,
                                                                        default=default)

        return cls._raw_create_table_sql(db_table, cols_def, order_by_clause,
                                         primary_by_clause=primary_by_clause,
                                         sample_clause=sample_clause,
                                         engine_type=engine_type, on_cluster=on_cluster,
                                         partition_by_clause=partition_by_clause, ttl=ttl, settings=settings)


class CreateTableUtils(CreateTableFromInfoUtils, CreateTableFromSQLUtils):
    # TODO test merge function for create table
    #
    @classmethod
    def _create(cls, db: str, table: str, df_or_sql_or_dict: (pd.DataFrame, str, dict),
                key_cols: (list, tuple),
                extra_format_dict: (dict, None) = None,
                primary_key_cols: (list, tuple, None) = None,
                sample_by_cols: (list, tuple, None) = None,
                partition_by_cols: (list, tuple, None) = None,
                settings: str = "SETTINGS index_granularity = 8192",
                engine_type: str = 'ReplacingMergeTree',
                on_cluster: str = '',
                ttl: str = '',
                other: str = '',
                ):
        if isinstance(df_or_sql_or_dict, dict):
            db_table = f"{db}.{table}"
            var_dict = df_or_sql_or_dict
            sql = cls.creator_info(db_table, var_dict,
                                   order_by_cols=key_cols,
                                   sample_by_cols=sample_by_cols,  #: (list, tuple, None)
                                   partition_by_cols=partition_by_cols,  # : (list, tuple, None)
                                   primary_by_cols=primary_key_cols,  # : (list, tuple, None)
                                   on_cluster=on_cluster,
                                   engine_type=engine_type,
                                   ttl=ttl, settings=settings)
        elif isinstance(df_or_sql_or_dict, (pd.DataFrame, str)):

            sql = cls.creator_sql(db, table, df_or_sql_or_dict, key_cols,
                                  extra_format_dict=extra_format_dict,
                                  sample_by_cols=sample_by_cols,  #: (list, tuple, None)
                                  partition_by_cols=partition_by_cols,  # : (list, tuple, None)
                                  primary_key_cols=primary_key_cols,  # : (list, tuple, None)
                                  on_cluster=on_cluster,
                                  engine_type=engine_type,
                                  settings=settings,
                                  other=ttl + ' ' + other)

        else:
            raise ValueError(f'unsupported type of df_or_sql_or_dict: {type(df_or_sql_or_dict)}')
        return sql


class TableEngineCreator(ClickHouseTableNode, CreateTableUtils):
    def __init__(self, conn_str: (str, None) = None, **kwarg):
        super(TableEngineCreator, self).__init__(conn_str=conn_str, **kwarg)

    def _create_view(self, db: str, table: str, sql: str,
                     key_cols: list,
                     primary_key_cols=None,
                     sample_by_cols=None,
                     engine_type: str = 'view',
                     extra_format_dict: (dict, None) = None,
                     partition_by_cols: (str, None) = None,
                     **kwargs):
        if not self._check_exists(f'{db}.{table}', mode='table'):
            pass
        else:
            raise ClickHouseTableExistsError(f'{db}.{table} is exists!')
        if engine_type.lower() == 'view':
            create_sql = f'create view if not exists {db}.{table} as {sql}'
            self.query(create_sql)
        else:
            raise ValueError(f'current not support material view : {engine_type}')

    # def create_table_if_not_exists(self, db: str, table: str, sql: (str, pd.DataFrame, None),
    #                                key_cols: list,
    #                                primary_key_cols=None, sample_expr=None,
    #                                engine_type: str = 'ReplacingMergeTree',
    #                                extra_format_dict: (dict, None) = None, partitions_expr: (str, None) = None):
    #     """
    #
    #     :param db:
    #     :param table:
    #     :param sql:
    #     :param key_cols:
    #     :param primary_key_cols:
    #     :param sample_expr:
    #     :param engine_type:
    #     :param extra_format_dict:
    #     :param partitions_expr:
    #     :return:
    #     """
    #
    #     self._create_table(db, table, sql, key_cols,
    #                            primary_key_cols=primary_key_cols,
    #                            sample_expr=sample_expr,
    #                            engine_type=engine_type,
    #                            extra_format_dict=extra_format_dict,
    #                            partitions_expr=partitions_expr)

    def create(self,
               db: str,
               table: str,
               sql: (str, pd.DataFrame, dict),
               key_cols: list,
               engine_type: str = 'ReplacingMergeTree',
               primary_key_cols: (list, tuple, None) = None,
               sample_by_cols: (list, tuple, None) = None,
               extra_format_dict: (dict, None) = None,
               partition_by_cols: (list, tuple, None) = None,
               settings: str = "SETTINGS index_granularity = 8192",
               on_cluster: str = '',
               ttl: str = '',
               other: str = '',
               check: bool = True,
               execute: bool = False):

        # if self._check_exists(f'{db}.{table}', mode='table'):
        #     raise ClickHouseTableExistsError(f'{db}.{table} is exists!')
        # else:
        create_sql = self._create(db, table, sql,
                                  key_cols,
                                  extra_format_dict=extra_format_dict,
                                  primary_key_cols=primary_key_cols,
                                  sample_by_cols=sample_by_cols,
                                  partition_by_cols=partition_by_cols,
                                  settings=settings,
                                  engine_type=engine_type,
                                  on_cluster=on_cluster,
                                  ttl=ttl,
                                  other=other,
                                  )
        if check:
            exist_status = self._check_exists(db_table=f'{db}.{table}')

            if exist_status:
                raise ClickHouseTableExistsError(f'{db}.{table} is exists!')
        if execute:
            self.query(create_sql)
        else:
            return create_sql


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
if __name__ == '__main__':
    pass
