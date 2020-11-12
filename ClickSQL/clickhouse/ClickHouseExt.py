# coding=utf-8
import pandas as pd
import re

from ClickSQL.clickhouse.ClickHouseCreate import TableEngineCreator

from ClickSQL.errors import ClickHouseTableExistsError, ParameterTypeError


class ClickHouseTableNodeExt(TableEngineCreator):
    def __init__(self, conn_str: (str, dict, None) = None, **kwarg):
        super(ClickHouseTableNodeExt, self).__init__(conn_str=conn_str, **kwarg)

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



if __name__ == '__main__':
    pass
