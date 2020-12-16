# coding=utf-8
class GroupSQLUtils(object):
    @staticmethod
    def group_top(sql: str, by: (str, list, tuple), top: int = 5, cols: (str, None) = None):
        if isinstance(by, str):
            by = [by]
        if cols is None:
            cols = '*'
        gt_sql = f"select {cols} from ({sql})  limit {top} by {','.join(by)} "
        return gt_sql

    # group table
    @staticmethod
    def group_by(db_table_or_sql: str,
                 by: (str, list, tuple),
                 apply_func: (list,),
                 having: (list, tuple, None) = None):
        if isinstance(by, str):
            by = [by]
            group_by_clause = f"group by {by}"
        elif isinstance(by, (list, tuple)):
            group_by_clause = f"group by ({','.join(by)})"
        else:
            raise ValueError(f'by only accept str list tuple! but get {type(by)}')
        # db_table_or_sql = sql
        if having is None:
            having_clause = ''
        elif isinstance(having, (list, tuple)):
            having_clause = 'having ' + " and ".join(having)
        else:
            raise ValueError(f'having only accept list,tuple,None! but get {type(having)}')
        sql = f"select  {','.join(by + apply_func)}  from ({db_table_or_sql}) {group_by_clause} {having_clause} "
        # if execute:
        #     self.operator(sql)
        # else:
        return sql
