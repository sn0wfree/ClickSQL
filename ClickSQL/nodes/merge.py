# coding=utf-8


class MergeSQLUtils(object):
    # merge table
    @staticmethod
    def _merge(first,
               seconds: str,
               using: (list, str, tuple),
               cols: (list, str, None) = None,
               join_type='all full join',
               # cols: list,
               #  sample: (int, float, None) = None,
               #  array_join: (list, None) = None,
               #  join: (dict, None) = None,
               #  prewhere: (list, None) = None,
               #  where: (list, None) = None,
               #  having: (list, None) = None,
               #  group_by: (list, None) = None,
               #  order_by: (list, None) = None,
               #  limit_by: (dict, None) = None,
               #  limit: (int, None) = None
               ) -> str:
        from ClickSQL.clickhouse.ClickHouseCreate import SQLBuilder
        # self._complex = True
        if isinstance(using, (list, tuple)):
            using = ','.join(using)

        join = {'type': join_type, 'USING': using, 'sql': str(seconds)}
        sql = SQLBuilder.select(str(first), cols, join=join, limit=None)
        # if execute:
        #     return self.operator(sql)
        # else:
        return sql


if __name__ == '__main__':
    pass
