# coding=utf-8
"""
GROUP BY SQL Utilities for ClickHouse.

This module provides SQL generation utilities for GROUP BY operations
in ClickHouse, including various aggregation functions.

Supported Operations:
- Basic GROUP BY
- GROUP BY with TOP N results
- HAVING clause
- Various aggregation functions (count, sum, avg, max, min, etc.)
- Advanced aggregations (countDistinct, groupArray, quantile, topK, etc.)

Example:
    >>> from ClickSQL.nodes.groupby import GroupSQLUtils
    >>> sql = GroupSQLUtils.group_by("SELECT * FROM table", by=['col1'], apply_func=['sum(col2)'])
    >>> sql = GroupSQLUtils.count("SELECT * FROM table", 'col1')
    >>> sql = GroupSQLUtils.quantile("SELECT * FROM table", 'col1', 0.5)
"""


class GroupSQLUtils:
    """
    SQL generation utilities for GROUP BY operations.
    
    This class provides static methods to generate SQL queries with
    various GROUP BY and aggregation patterns commonly used in ClickHouse.
    """

    @staticmethod
    def group_top(sql: str, by: (str, list, tuple), top: int = 5, cols: (str, None) = None):
        """
        Generate SELECT with LIMIT BY for top N results per group.
        
        Args:
            sql: Source SQL query or table name
            by: Column(s) to group by
            top: Number of results per group
            cols: Columns to select (None for all)
            
        Returns:
            SQL string with LIMIT BY clause
            
        Example:
            >>> GroupSQLUtils.group_top("SELECT * FROM orders", by=['customer_id'], top=10)
            "select * from (SELECT * FROM orders)  limit 10 by customer_id "
        """
        if isinstance(by, str):
            by = [by]
        if cols is None:
            cols = '*'
        gt_sql = f"select {cols} from ({sql})  limit {top} by {','.join(by)} "
        return gt_sql

    @staticmethod
    def group_by(db_table_or_sql: str,
                 by: (str, list, tuple),
                 apply_func: (list,),
                 having: (list, tuple, None) = None):
        """
        Generate GROUP BY query with aggregation functions.
        
        Args:
            db_table_or_sql: Source table or SQL query
            by: Column(s) to group by
            apply_func: List of aggregation functions (e.g., ['sum(col)', 'count(*)'])
            having: Optional HAVING conditions
            
        Returns:
            SQL string with GROUP BY clause
            
        Example:
            >>> GroupSQLUtils.group_by("orders", by=['customer_id'], apply_func=['sum(amount)'])
            "select  customer_id,sum(amount)  from (orders) group by (customer_id)  "
        """
        if isinstance(by, str):
            by = [by]
            group_by_clause = f"group by {by}"
        elif isinstance(by, (list, tuple)):
            group_by_clause = f"group by ({','.join(by)})"
        else:
            raise ValueError(f'by only accept str list tuple! but get {type(by)}')
        
        if having is None:
            having_clause = ''
        elif isinstance(having, (list, tuple)):
            having_clause = 'having ' + " and ".join(having)
        else:
            raise ValueError(f'having only accept list,tuple,None! but get {type(having)}')
        
        sql = f"select  {','.join(by + apply_func)}  from ({db_table_or_sql}) {group_by_clause} {having_clause} "
        return sql

    @staticmethod
    def count(sql: str, by: (str, list), alias: str = 'cnt'):
        """
        Generate count aggregation.
        
        Args:
            sql: Source SQL or table
            by: Column(s) to group by
            alias: Alias for count column
            
        Returns:
            SQL string with count
        """
        if isinstance(by, str):
            by = [by]
        func = f"count() as {alias}"
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def sum(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate sum aggregation.
        
        Args:
            sql: Source SQL or table
            col: Column to sum
            alias: Alias for sum column (default: 'sum_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with sum
        """
        if alias is None:
            alias = f"sum_{col}"
        func = f"sum({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def avg(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate average aggregation.
        
        Args:
            sql: Source SQL or table
            col: Column to average
            alias: Alias for avg column (default: 'avg_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with average
        """
        if alias is None:
            alias = f"avg_{col}"
        func = f"avg({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def max(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate max aggregation.
        
        Args:
            sql: Source SQL or table
            col: Column to find max
            alias: Alias for max column (default: 'max_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with max
        """
        if alias is None:
            alias = f"max_{col}"
        func = f"max({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def min(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate min aggregation.
        
        Args:
            sql: Source SQL or table
            col: Column to find min
            alias: Alias for min column (default: 'min_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with min
        """
        if alias is None:
            alias = f"min_{col}"
        func = f"min({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def countDistinct(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate countDistinct aggregation (unique count).
        
        Args:
            sql: Source SQL or table
            col: Column to count distinct
            alias: Alias for count column (default: 'uniq_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with count distinct
        """
        if alias is None:
            alias = f"uniq_{col}"
        func = f"uniqExact({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def groupArray(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None, limit: (int, None) = None):
        """
        Generate groupArray aggregation (collect values into array).
        
        Args:
            sql: Source SQL or table
            col: Column to collect
            alias: Alias for array column (default: 'arr_<col>')
            by: Column(s) to group by
            limit: Optional limit on array size
            
        Returns:
            SQL string with groupArray
        """
        if alias is None:
            alias = f"arr_{col}"
        func_expr = f"groupArray({col})" if limit is None else f"groupArray({limit})({col})"
        func = f"{func_expr} as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def quantile(sql: str, col: str, q: float = 0.5, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate quantile aggregation.
        
        Args:
            sql: Source SQL or table
            col: Column to calculate quantile
            q: Quantile value (0-1), default 0.5 for median
            alias: Alias for quantile column (default: 'q<q>_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with quantile
        """
        if alias is None:
            alias = f"q{q}_{col}"
        func = f"quantile({q})({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def topK(sql: str, col: str, k: int = 10, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate topK aggregation (most frequent values).
        
        Args:
            sql: Source SQL or table
            col: Column to find top K
            k: Number of top values to return
            alias: Alias for topK column (default: 'top_{col}')
            by: Column(s) to group by
            
        Returns:
            SQL string with topK
        """
        if alias is None:
            alias = f"top_{col}"
        func = f"topK({k})({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def stddev(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate standard deviation aggregation.
        
        Args:
            sql: Source SQL or table
            col: Column to calculate stddev
            alias: Alias for stddev column (default: 'std_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with stddev
        """
        if alias is None:
            alias = f"std_{col}"
        func = f"stddevPop({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])

    @staticmethod
    def any(sql: str, col: str, alias: (str, None) = None, by: (str, list) = None):
        """
        Generate any aggregation (get first value).
        
        Args:
            sql: Source SQL or table
            col: Column to get first value
            alias: Alias for any column (default: 'any_<col>')
            by: Column(s) to group by
            
        Returns:
            SQL string with any
        """
        if alias is None:
            alias = f"any_{col}"
        func = f"any({col}) as {alias}"
        by = by or []
        if isinstance(by, str):
            by = [by]
        return GroupSQLUtils.group_by(sql, by=by, apply_func=[func])


if __name__ == '__main__':
    print(GroupSQLUtils.group_by("orders", by=['customer_id'], apply_func=['sum(amount)', 'count()']))
    print(GroupSQLUtils.count("orders", 'product_id'))
    print(GroupSQLUtils.quantile("events", 'value', 0.95, by=['category']))
    print(GroupSQLUtils.topK("clicks", 'user_id', 5))
