# coding=utf-8
"""
ClickHouse Extended Features Module.

This module provides extended functionality for ClickHouse operations including:
- Advanced query building
- Table optimization
- Projection management
- System logs access
- Table structure caching

Example:
    >>> from ClickSQL import ClickHouseTableNodeExt
    >>> node = ClickHouseTableNodeExt("clickhouse://user:pass@host:8123/db")
    >>> node.optimize_table('db', 'table', final=True)
    >>> projections = node.list_projections('db', 'table')
"""

import pandas as pd
import re
from collections import namedtuple, ChainMap
from functools import lru_cache
from ClickSQL.clickhouse.ClickHouseCreate import TableEngineCreator

from ClickSQL.errors import ClickHouseTableExistsError, ParameterTypeError

factor_parameters = ('dt', 'code', 'value', 'fid')
ft_node = namedtuple('factortable', factor_parameters)


class ClickHouseTableNodeExt(TableEngineCreator):
    """
    Extended ClickHouse Table Node with advanced features.
    
    This class extends ClickHouseTableNode with additional functionality
    including query building, table optimization, and projection management.
    
    Attributes:
        _table_columns_cache: Internal cache for table column information
        
    Example:
        >>> node = ClickHouseTableNodeExt("clickhouse://user:pass@host:8123/db")
        >>> node.optimize_table('db', 'table', final=True)
    """

    def __init__(self, conn_str: (str, dict, None) = None, **kwarg):
        """
        Initialize extended ClickHouse table node.
        
        Args:
            conn_str: Connection string or dictionary
            **kwarg: Additional connection parameters
        """

    def __init__(self, conn_str: (str, dict, None) = None, **kwarg):
        super(ClickHouseTableNodeExt, self).__init__(conn_str=conn_str, **kwarg)
        self._src = conn_str
        self.db_table = self._para.database
        self._table_columns_cache = {}

    def _get_table_columns_cached(self, db_table: str):
        """
        Get table columns with caching.
        
        Retrieves column names for a table and caches the result to avoid
        repeated DESCRIBE TABLE queries.
        
        Args:
            db_table: Database and table name (db.table format)
            
        Returns:
            List of column names
        """
        if db_table not in self._table_columns_cache:
            self._table_columns_cache[db_table] = self.query(f"desc {db_table}")['name'].values.tolist()
        return self._table_columns_cache[db_table]

    def clear_table_cache(self, db_table: str = None):
        """
        Clear table columns cache.
        
        Args:
            db_table: Specific table to clear, or None to clear all
        """
        if db_table:
            self._table_columns_cache.pop(db_table, None)
        else:
            self._table_columns_cache.clear()

    @staticmethod
    def __extend_dict_value__(conditions: (dict, ChainMap)):
        """
        Extend dict values for SQL filter conditions.
        
        Args:
            conditions: Dictionary of filter conditions
            
        Yields:
            Filter condition strings
        """
        for s in conditions.values():
            if isinstance(s, str):
                yield s
            elif isinstance(s, (tuple, list)):
                for s_sub in s:
                    yield s_sub
            else:
                raise ValueError('filter settings get wrong type! only accept string and tuple of string')

    def explain(self, sql: str):
        """
        Explain SQL query execution plan.
        
        Args:
            sql: SQL query to explain
            
        Returns:
            Query result with execution plan
        """
        return self.query(f'explain {sql}')

    @staticmethod
    def __obtain_other_filter__(other_filters):
        exits_keys = []
        for k, v in other_filters.items():
            if k in exits_keys:
                raise ValueError(f'found duplicated key: {k}')
            exits_keys.append(k)
            if isinstance(v, dict):
                yield v
            elif isinstance(v, (str, tuple)):
                yield {k: v}
            else:
                raise ValueError('filter settings get wrong type! only accept string and tuple of string')

    def get_sql(self, db_table: str, cols: (tuple, None, list) = None,
                order_by_cols: (list, tuple, None) = None,
                data_filter: dict = {}, include_filter=True,
                limit: (None, int, str) = None,
                use_cache: bool = True,
                **other_filters):
        """
        Build SELECT SQL query with filters and options.
        
        Args:
            db_table: Database and table name (db.table format)
            cols: List of columns to select, None for all, ['*'] expands to all
            order_by_cols: List of ORDER BY clauses, e.g., ['col1 asc', 'col2 desc']
            data_filter: Dictionary of column:value filters
            include_filter: Whether to include filter columns in SELECT
            limit: Row limit (int or 'LIMIT n' string)
            use_cache: Whether to use cached table columns
            **other_filters: Additional filter conditions as keyword arguments
            
        Returns:
            Generated SQL query string
        """
        if cols is None:
            cols = ['*']
        elif len(cols) == 0:
            cols = ['*']

        if '*' in cols:
            cols = list(cols)
            cols.pop(cols.index('*'))
            if use_cache:
                cols.extend(self._get_table_columns_cached(db_table))
            else:
                cols.extend(self.query(f"desc {db_table}")['name'].values.tolist())
        conditions = ChainMap(data_filter, *list(self.__obtain_other_filter__(other_filters)))
        filter_yield = self.__extend_dict_value__(conditions)
        if include_filter:
            cols = sorted(set(list(cols) + list(conditions.keys())))
        else:
            cols = sorted(set(cols))
        if order_by_cols is None:
            order_by_clause = ''
        elif len(order_by_cols) > 1:
            order_by_clause = f" order by ({','.join(order_by_cols)})"
        elif len(order_by_cols) == 1:
            order_by_clause = f" order by {','.join(order_by_cols)}"
        else:
            raise ValueError('order_by_cols get wrong length')
        if limit is None:
            limit_clause = ''
        elif isinstance(limit, int):
            limit_clause = f"limit {limit}"
        elif isinstance(limit, str) and limit.strip(' ').lower().startswith('limit'):
            limit_clause = limit
        else:
            raise ValueError(f'limit parameter got wrong type! only accept str,int or None, but got {type(limit)}')
        where_clause = ' and '.join(sorted(set(['1'] + list(filter_yield))))
        sql = f"select {','.join(cols)} from {db_table} where {where_clause} {order_by_clause} {limit_clause} "
        return sql

    def _execute(self, sql: str, **kwargs):
        return self.query(sql, **kwargs)
    
    def optimize_table(self, db: str, table: str, final: bool = False, deduplicate: bool = False):
        """
        Optimize table storage.
        
        :param db: Database name
        :param table: Table name
        :param final: If True, optimize to final format
        :param deduplicate: If True, deduplicate merge tree parts
        :return: Query result
        """
        parts = []
        if final:
            parts.append('FINAL')
        if deduplicate:
            parts.append('DEDUPLICATE')
        
        sql = f"OPTIMIZE TABLE {db}.{table}"
        if parts:
            sql += " " + " ".join(parts)
        
        return self.query(sql)
    
    def list_projections(self, db: str, table: str):
        """
        List all projections for a table.
        
        :param db: Database name
        :param table: Table name
        :return: DataFrame with projection info
        """
        sql = f"""
        SELECT name, format, partitioning_key, sorting_key, 
               primary_key, storage_policy, ttl
        FROM system.projections
        WHERE database = '{db}' AND table = '{table}'
        """
        return self.query(sql)
    
    def create_projection(self, db: str, table: str, projection_name: str, 
                         select_query: str, partition_by: str = None, order_by: str = None):
        """
        Create a projection for a table.
        
        :param db: Database name
        :param table: Table name
        :param projection_name: Name for the projection
        :param select_query: SELECT query for projection
        :param partition_by: PARTITION BY clause
        :param order_by: ORDER BY clause
        :return: Query result
        """
        sql = f"""
        ALTER TABLE {db}.{table}
        ADD PROJECTION {projection_name}
        ({select_query})
        """
        
        if partition_by:
            sql += f" PARTITION BY {partition_by}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        
        return self.query(sql)
    
    def drop_projection(self, db: str, table: str, projection_name: str):
        """
        Drop a projection from a table.
        
        :param db: Database name
        :param table: Table name
        :param projection_name: Name of projection to drop
        :return: Query result
        """
        sql = f"ALTER TABLE {db}.{table} DROP PROJECTION {projection_name}"
        return self.query(sql)
    
    def system_logs(self, log_type: str = 'query_log', limit: int = 100):
        """
        Query system logs.
        
        :param log_type: Type of log (query_log, part_log, metric_log, etc.)
        :param limit: Number of rows to return
        :return: DataFrame with log entries
        """
        valid_logs = ['query_log', 'part_log', 'metric_log', 'trace_log', 
                     'error_log', 'text_log', 'asynchronous_metric_log']
        if log_type not in valid_logs:
            raise ValueError(f"Invalid log_type. Must be one of: {valid_logs}")
        
        sql = f"SELECT * FROM system.{log_type} ORDER BY event_time DESC LIMIT {limit}"
        return self.query(sql)


if __name__ == '__main__':
    pass
