# coding=utf-8
"""
MERGE/JOIN SQL Utilities for ClickHouse.

This module provides SQL generation utilities for MERGE and JOIN operations
in ClickHouse.

Supported Operations:
- Basic table merging with USING clause
- Various join types (INNER, LEFT, RIGHT, FULL, CROSS)
- Support for subqueries and table references

Example:
    >>> from ClickSQL.nodes.merge import MergeSQLUtils
    >>> sql = MergeSQLUtils.merge("table1", "table2", using=['id'])
    >>> sql = MergeSQLUtils.merge("table1", "SELECT * FROM table2", using=['user_id'], join_type='inner join')
"""


class MergeSQLUtils:
    """
    SQL generation utilities for MERGE/JOIN operations.
    
    This class provides static methods to generate SQL queries with
    various JOIN patterns commonly used in ClickHouse.
    """

    JOIN_TYPES = {
        'inner': 'INNER JOIN',
        'left': 'LEFT JOIN',
        'right': 'RIGHT JOIN',
        'full': 'FULL OUTER JOIN',
        'cross': 'CROSS JOIN',
        'all': 'ALL JOIN',
        'any': 'ANY JOIN',
        'asof': 'ASOF JOIN'
    }

    @staticmethod
    def _merge(first,
               seconds: str,
               using: (list, str, tuple),
               cols: (list, str, None) = None,
               join_type: str = 'all full join',
               ) -> str:
        """
        Generate MERGE/JOIN query for combining tables.
        
        Args:
            first: First table or query (source node or string)
            seconds: Second table or SQL query to join
            using: Column(s) to join on (USING clause)
            cols: Columns to select (None for all columns)
            join_type: Type of join (default: 'all full join')
            
        Returns:
            SQL string with JOIN clause
            
        Example:
            >>> MergeSQLUtils.merge("orders", "customers", using=['customer_id'])
            >>> MergeSQLUtils.merge("SELECT * FROM a", "SELECT * FROM b", using=['id'], join_type='inner join')
        """
        from ClickSQL.clickhouse.ClickHouseCreate import SQLBuilder
        
        if isinstance(using, (list, tuple)):
            using = ','.join(using)

        join = {'type': join_type, 'USING': using, 'sql': str(seconds)}
        sql = SQLBuilder.select(str(first), cols, join=join, limit=None)
        return sql

    @staticmethod
    def inner_join(first, second: str, using: (list, str), cols: (list, str, None) = None) -> str:
        """
        Generate INNER JOIN query.
        
        Args:
            first: First table or query
            second: Second table or SQL query
            using: Column(s) to join on
            cols: Columns to select
            
        Returns:
            SQL string with INNER JOIN
        """
        return MergeSQLUtils._merge(first, second, using=using, join_type='INNER JOIN', cols=cols)

    @staticmethod
    def left_join(first, second: str, using: (list, str), cols: (list, str, None) = None) -> str:
        """
        Generate LEFT JOIN query.
        
        Args:
            first: First table or query
            second: Second table or SQL query
            using: Column(s) to join on
            cols: Columns to select
            
        Returns:
            SQL string with LEFT JOIN
        """
        return MergeSQLUtils._merge(first, second, using=using, join_type='LEFT JOIN', cols=cols)

    @staticmethod
    def right_join(first, second: str, using: (list, str), cols: (list, str, None) = None) -> str:
        """
        Generate RIGHT JOIN query.
        
        Args:
            first: First table or query
            second: Second table or SQL query
            using: Column(s) to join on
            cols: Columns to select
            
        Returns:
            SQL string with RIGHT JOIN
        """
        return MergeSQLUtils._merge(first, second, using=using, join_type='RIGHT JOIN', cols=cols)

    @staticmethod
    def full_join(first, second: str, using: (list, str), cols: (list, str, None) = None) -> str:
        """
        Generate FULL OUTER JOIN query.
        
        Args:
            first: First table or query
            second: Second table or SQL query
            using: Column(s) to join on
            cols: Columns to select
            
        Returns:
            SQL string with FULL OUTER JOIN
        """
        return MergeSQLUtils._merge(first, second, using=using, join_type='FULL OUTER JOIN', cols=cols)

    @staticmethod
    def asof_join(first, second: str, using: (list, str), cols: (list, str, None) = None) -> str:
        """
        Generate ASOF JOIN query (for time-series data).
        
        Args:
            first: First table or query
            second: Second table or SQL query
            using: Column(s) to join on (must be ordered)
            cols: Columns to select
            
        Returns:
            SQL string with ASOF JOIN
            
        Note:
            ASOF JOIN requires ClickHouse 19.x+ and sorted data
        """
        return MergeSQLUtils._merge(first, second, using=using, join_type='ASOF LEFT JOIN', cols=cols)


if __name__ == '__main__':
    print(MergeSQLUtils.inner_join("orders", "customers", using=['customer_id']))
    print(MergeSQLUtils.left_join("a", "SELECT * FROM b", using=['id']))
    print(MergeSQLUtils.asof_join("events", "prices", using=['timestamp']))
