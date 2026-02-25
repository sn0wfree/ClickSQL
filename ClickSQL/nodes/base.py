# coding=utf-8
"""
Nodes Module - Query Node Classes for ClickSQL.

This module provides node-based query abstractions for ClickHouse databases,
including factor table nodes, group by operations, merge operations, and
data update utilities.

Classes:
    - BaseSingleQueryBaseNode: Base class for single query operations
    - BaseSingleFactorTableNode: Factor table node for quantitative analysis
    - DelayTasks: Queue for delayed SQL execution
    - GroupSQLUtils: SQL generation utilities for GROUP BY operations
    - MergeSQLUtils: SQL generation utilities for MERGE/JOIN operations
    - UpdateSQLUtils: SQL generation utilities for data updates

Example:
    >>> from ClickSQL import BaseSingleFactorTableNode
    >>> node = BaseSingleFactorTableNode("clickhouse://user:pass@host:8123/db.table")
    >>> result = node.fetch(100)
    >>> node.save_to('target_table')
"""

import copy
import warnings
from collections import deque
from functools import wraps
from ClickSQL.pool.pooled_ch import PooledClickHouseTableNodeExt
from ClickSQL.clickhouse.ClickHouseExt import ClickHouseTableNodeExt
from ClickSQL.errors import ClickHouseTableNotExistsError
from ClickSQL.nodes.groupby import GroupSQLUtils
from ClickSQL.nodes.merge import MergeSQLUtils

complex_sql_select_count = 4


class DelayTasks(deque):
    """
    Queue for delayed SQL execution.
    
    This class extends deque to allow queuing multiple SQL statements
    for later execution. Useful for batch operations.
    
    Attributes:
        _query: Reference to the query execution function
        
    Example:
        >>> tasks = DelayTasks(node.operator)
        >>> tasks.append("INSERT INTO ...")
        >>> tasks.append("INSERT INTO ...")
        >>> results = tasks.run()  # Execute all queued queries
    """
    
    def __init__(self, _query, *args, **kwargs):
        super(DelayTasks, self).__init__(*args, **kwargs)
        self._query = _query

    def run(self, no_yield=False):
        """
        Execute all queued SQL statements.
        
        Args:
            no_yield: If True, return list; if False, return generator
            
        Returns:
            List or generator of query results
        """
        result = (self._query(sql) for sql in self)
        return list(result) if no_yield else result


class UpdateSQLUtils:
    """
    SQL generation utilities for data update operations.
    
    This class provides static methods to generate SQL for
    full and incremental data updates between tables.
    """
    
    @staticmethod
    def full_update(src_node, dst_db_table: str, fid_col: str = 'fid', dt_col: str = 'dt') -> str:
        """
        Generate SQL for full table update (replace all data).
        
        Args:
            src_node: Source BaseSingleQueryBaseNode
            dst_db_table: Destination table (db.table format)
            fid_col: Factor ID column name
            dt_col: Date column name
            
        Returns:
            SQL string for full update
            
        Example:
            >>> sql = UpdateSQLUtils.full_update(src_node, 'target.target_table')
            >>> node.operator(sql)
        """
        src_sql = src_node.__sql__
        sql = f"""
        INSERT INTO {dst_db_table}
        {src_sql}
        """
        return sql
    
    @staticmethod
    def incremental_update(src_node, dst_db_table: str, fid_col: str = 'fid', 
                          dt_col: str = 'dt', dt_max: str = None) -> str:
        """
        Generate SQL for incremental update (append new data).
        
        Args:
            src_node: Source BaseSingleQueryBaseNode
            dst_db_table: Destination table (db.table format)
            fid_col: Factor ID column name
            dt_col: Date column name
            dt_max: Maximum date to filter (optional)
            
        Returns:
            SQL string for incremental update
            
        Example:
            >>> sql = UpdateSQLUtils.incremental_update(src_node, 'target.target_table', dt_max='2024-01-01')
        """
        src_sql = src_node.__sql__
        
        if dt_max:
            where_clause = f"WHERE {dt_col} > '{dt_max}'"
        else:
            where_clause = ""
        
        sql = f"""
        INSERT INTO {dst_db_table}
        SELECT * FROM (
            {src_sql}
        ) {where_clause}
        """
        return sql


class BaseSingleQueryBaseNode(object):
    """
    Base class for single query operations on ClickHouse tables.
    
    This class provides a high-level abstraction for querying ClickHouse tables,
    with support for delayed execution, data updates, and table operations.
    
    Attributes:
        operator: ClickHouseTableNodeExt or PooledClickHouseTableNodeExt instance
        db: Database name
        table: Table name
        db_table: Full database.table name
        src: Source connection string
        delay_tasks: Queue for delayed SQL execution
        
    Example:
        >>> node = BaseSingleQueryBaseNode("clickhouse://user:pass@host:8123/db.table")
        >>> df = node.fetch(1000)
        >>> sql = node.groupby(['category'], apply_func=['count()'])
    """
    
    __Name__ = "基础因子库单因子基类"
    __slots__ = (
        'operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', 'status', '_INFO', 'depend_tables',
        '_fid_ck', '_dt_max_1st', '_execute', '_no_self_update', 'delay_tasks', 'src'
    )

    def __init__(self, src: str, db_table: (None, str) = None, info=None, pool=False, **kwargs):
        """
        Initialize base query node.
        
        Args:
            src: ClickHouse connection string
            db_table: Database and table name (db.table format), uses src default if None
            info: Optional metadata/info
            pool: Whether to use connection pool
            **kwargs: Query parameters:
                - cols: Columns to select
                - order_by_cols: ORDER BY columns
                - data_filter: WHERE conditions
                - include_filter: Include filter columns in SELECT
                - limit: Row limit
        """
        if pool:
            self.operator = PooledClickHouseTableNodeExt(src)
        else:
            self.operator = ClickHouseTableNodeExt(src)
        if db_table is None:
            src_db_table = self.operator.db_table
            self.db_table, self.db, self.table = self._db_split(src_db_table)
        elif isinstance(db_table, str):
            self.db_table, self.db, self.table = self._db_split(db_table)
        else:
            raise ValueError('db_table only accept str!')
        self.src = src
        self._kwargs = self._raw_kwargs = kwargs
        self.status = 'SQL'
        self._INFO = info
        self.delay_tasks = DelayTasks(self.operator)

    @staticmethod
    def _db_split(src_db_table):
        """
        Split database.table string into components.
        
        Args:
            src_db_table: String in 'db.table' format
            
        Returns:
            Tuple of (db_table, db, table)
        """
        if isinstance(src_db_table, str):
            if '.' in src_db_table:
                db, table = src_db_table.split('.')
                return src_db_table, db, table
            else:
                return f"{src_db_table}.None", src_db_table, 'None'
        else:
            raise ValueError('db_table parameter get wrong type! only accept str')

    @wraps(ClickHouseTableNodeExt.insert_df)
    def insert_df(self, *args, **kwargs):
        """Insert DataFrame into table."""
        self.operator.insert_df(*args, **kwargs)

    @wraps(ClickHouseTableNodeExt.create)
    def create(self, *args, **kwargs):
        """Create table."""
        return self.operator.create(*args, **kwargs)

    def _update(self, **kwargs):
        """
        Update query parameters/settings.
        
        Args:
            **kwargs: Parameters to update
        """
        self._kwargs.update(kwargs)

    def __str__(self):
        """Return SQL representation."""
        return self.__sql__

    def __len__(self) -> int:
        """
        Get row count of the query result.
        
        Returns:
            Number of rows
        """
        sql = f"select count(1) as rows from ({self.__sql__})"
        rows = self.operator(sql)['rows'].values[0]
        return rows

    @property
    def __sql__(self):
        """
        Generate SQL query from current parameters.
        
        Returns:
            SQL query string
        """
        return self.operator.get_sql(db_table=self.db_table, **self._kwargs)

    def decorate(self, target_conn_args='conn'):
        """
        Decorator factory for injecting connection into functions.
        
        Args:
            target_conn_args: Name of the argument to inject connection into
            
        Returns:
            Decorator function
        """
        def afunc(func):
            def _afunc(*args, **kwargs):
                if target_conn_args in kwargs.keys():
                    if kwargs[target_conn_args] is None:
                        kwargs[target_conn_args] = self.operator
                    elif callable(kwargs[target_conn_args]):
                        pass
                    else:
                        raise KeyError(f'target variable:{target_conn_args} had been setup into other value! ')
                else:
                    raise KeyError(f'cannot find target variable:{target_conn_args} ')
                return func(*args, **kwargs)
            return _afunc
        return afunc

    def __call__(self, *sql, **kwargs):
        """
        Execute SQL or queue for delayed execution.
        
        Args:
            *sql: SQL string(s) to execute
            delay: If True, queue SQL for later execution
            
        Returns:
            Query results or None if queued
        """
        if 'delay' in kwargs.keys():
            if kwargs['delay']:
                if isinstance(sql, str):
                    self.delay_tasks.append(sql)
                else:
                    for s in sql:
                        self.delay_tasks.append(s)
            else:
                kwargs.pop('delay')
                return self.operator(*sql, **kwargs)
        else:
            return self.operator(*sql, **kwargs)

    @property
    def __factor_id__(self):
        """Get unique factor ID based on SQL hash."""
        return hash(self.__sql__)

    def __getitem__(self, key: (list, str)):
        """
        Execute query with selected columns.
        
        Args:
            key: Column name or list of column names
            
        Returns:
            Query result DataFrame
        """
        if isinstance(key, list):
            sql = f"select {','.join(key)} from ({self.__sql__})"
            return self.operator(sql)
        elif isinstance(key, str):
            sql = f"select {key} from ({self.__sql__})"
            return self.operator(sql)
        else:
            raise ValueError('key only accept list or str')

    @property
    def __system_tables__(self):
        """Get system table metadata."""
        sql = f"select total_rows,engine from system.tables where database ='{self.db}' and name='{self.table}'"
        res = self.operator(sql)
        return res

    def _detect_complex_sql(self, warn=False):
        """
        Detect potentially complex SQL queries.
        
        Args:
            warn: If True, issue warning; if False, raise ValueError
        """
        sql = self.__sql__.lower()
        if 'select' in sql and len(sql.split('select ')) >= complex_sql_select_count:
            if warn:
                warnings.warn('this sql may be a complex sql!')
            else:
                raise ValueError('this sql may be a complex sql!')

    @property
    def table_exist(self):
        """
        Check if table exists.
        
        Returns:
            True if table exists
        """
        return not self.__system_tables__.empty

    @property
    def table_engine(self):
        """
        Get table engine type.
        
        Returns:
            Engine name (e.g., 'MergeTree', 'Memory')
        """
        if self.table_exist:
            return self.__system_tables__['engine'].values[0]
        else:
            raise ClickHouseTableNotExistsError(f'{self.db_table} is not exists!')

    def fetch(self, num=1000, pattern=r'[\s]+limit[\s]+[0-9]+$', **kwargs):
        """
        Fetch first N rows from query.
        
        Args:
            num: Number of rows to fetch
            pattern: Regex pattern for LIMIT detection
            **kwargs: Additional query parameters
            
        Returns:
            Query result DataFrame
        """
        sql = self.__sql__
        self._detect_complex_sql()
        end_with_limit = self.operator._check_end_with_limit(sql, pattern=pattern)
        if end_with_limit:
            return self.operator(sql, **kwargs)
        else:
            try:
                return self.operator(sql + f' limit {num}', **kwargs)
            except Exception as e:
                return self.operator(sql, **kwargs)

    def fetch_all(self, **kwargs):
        """
        Fetch all rows from query.
        
        Args:
            **kwargs: Additional query parameters
            
        Returns:
            Query result DataFrame
        """
        self._detect_complex_sql()
        return self.operator(self.__sql__, **kwargs)

    @property
    def row_count(self):
        """Get row count."""
        return self.__len__()

    @property
    def shape(self):
        """
        Get shape (rows, columns).
        
        Returns:
            Tuple of (row_count, col_count)
        """
        return self.row_count, self.col_count

    @property
    def dtypes(self):
        """Get column data types."""
        sql = f"desc ({self.__sql__})"
        dtypes = self.operator(sql)
        return dtypes

    @property
    def columns(self):
        """Get column names."""
        return self.dtypes['name'].values.tolist()

    @property
    def col_count(self):
        """Get column count."""
        return self.dtypes.shape[0]

    @property
    def empty(self):
        """Check if result is empty."""
        return self.total_rows == 0

    @property
    def total_rows(self):
        """
        Get total rows in table.
        
        Returns:
            Row count from system.tables
        """
        temp = self.__system_tables__
        if not temp.empty:
            return temp['total_rows'].values[0]
        else:
            raise ClickHouseTableNotExistsError(f'{self.db_table} is not exists!')

    def nlargest(self, top: int, columns: list, execute: bool = True, extra_cols: (str, None) = None):
        """
        Get largest N rows by specified columns.
        
        Args:
            top: Number of rows
            columns: Columns to order by
            execute: If True, execute and return result; if False, return SQL
            extra_cols: Additional columns to include
            
        Returns:
            DataFrame or SQL string
        """
        by = [f'{c} desc' for c in columns]
        sql = GroupSQLUtils.group_top(self.__sql__, by=by, top=top, cols=extra_cols)
        if execute:
            return self.operator(sql)
        else:
            return sql

    def nsmallest(self, top: int, columns: list, execute: bool = True, extra_cols: (str, None) = None):
        """
        Get smallest N rows by specified columns.
        
        Args:
            top: Number of rows
            columns: Columns to order by
            execute: If True, execute and return result; if False, return SQL
            extra_cols: Additional columns to include
            
        Returns:
            DataFrame or SQL string
        """
        by = [f'{c} asc' for c in columns]
        sql = GroupSQLUtils.group_top(self.__sql__, by=by, top=top, cols=extra_cols)
        if execute:
            return self.operator(sql)
        else:
            return sql

    def groupby(self, by: (str, list, tuple), apply_func: (list,), having: (list, tuple, None) = None, execute=True):
        """
        Execute GROUP BY query.
        
        Args:
            by: Column(s) to group by
            apply_func: Aggregation functions
            having: HAVING conditions
            execute: If True, execute and return result; if False, return SQL
            
        Returns:
            DataFrame or SQL string
        """
        sql = GroupSQLUtils.group_by(self.__sql__, by=by, apply_func=apply_func, having=having)
        if execute:
            return self.operator(sql)
        else:
            return sql

    def merge(self, seconds, using: (list, str, tuple), join_type='all full join', cols: (list, str, None) = None,
              execute=True):
        """
        Execute MERGE/JOIN query.
        
        Args:
            seconds: Second table or SQL to join
            using: Column(s) to join on
            join_type: Type of join
            cols: Columns to select
            execute: If True, execute and return result; if False, return SQL
            
        Returns:
            DataFrame or SQL string
        """
        if seconds.lower().startswith('select'):
            pass
        else:
            seconds = f" select * from {seconds}"
        sql = MergeSQLUtils._merge(self, seconds, using=using, join_type=join_type, cols=cols)
        if execute:
            return self.operator(sql)
        else:
            return sql

    def __lshift__(self, src_db_table):
        """
        Load data from source table (<< operator).
        
        This operator loads/updates data from another table into this node's table.
        
        Args:
            src_db_table: Source table (string or BaseSingleQueryBaseNode)
            
        Returns:
            Tuple of (sql, update_status)
            
        Example:
            >>> target << source  # Load from source
        """
        fid_ck = self._fid_ck if hasattr(self, '_fid_ck') else 'fid'
        dt_max_1st = self._dt_max_1st if hasattr(self, '_dt_max_1st') else None
        execute = self._execute if hasattr(self, '_execute') else True
        no_self_update = self._no_self_update if hasattr(self, '_no_self_update') else True

        if isinstance(src_db_table, str):
            src_conn = copy.deepcopy(self.operator._src).replace(self.db_table, src_db_table)
            src_node = BaseSingleQueryBaseNode(src_conn, cols=['*'])
        elif isinstance(src_db_table, BaseSingleQueryBaseNode):
            src_node = src_db_table
        else:
            raise ValueError('src_db_table is not valid! please check!')

        if src_node.empty:
            raise ValueError(f'{src_db_table.db_table} is empty')
        
        if no_self_update and self.db_table == src_node.db_table and self.__factor_id__ == src_node.__factor_id__:
            dst = src_node.db_table
            src = self.db_table
            raise ValueError(
                f'Detect self-update process! these operator attempts to update data from {src} to {dst}')

        update_status = 'full' if self.empty else 'incremental'

        func = getattr(UpdateSQLUtils, f'{update_status}_update')
        sql = func(src_node, self.db_table, fid_col=fid_ck, dt_col=dt_max_1st)
        if execute:
            self.operator(sql)
        return sql, update_status

    def __rshift__(self, dst_db_table):
        """
        Save data to target table (>> operator).
        
        This operator saves this node's data to another table.
        
        Args:
            dst_db_table: Target table (string or BaseSingleQueryBaseNode)
            
        Returns:
            Tuple of (sql, update_status)
            
        Example:
            >>> source >> target  # Save to target
        """
        fid_ck = self._fid_ck if hasattr(self, '_fid_ck') else 'fid'
        dt_max_1st = self._dt_max_1st if hasattr(self, '_dt_max_1st') else None
        execute = self._execute if hasattr(self, '_execute') else True
        no_self_update = self._no_self_update if hasattr(self, '_no_self_update') else True
        
        if self.empty:
            raise ValueError(f'{self.db_table} is empty')

        if isinstance(dst_db_table, str):
            dst_conn = copy.deepcopy(self.operator._src)
            dst_conn = dst_conn.replace(self.db_table, dst_db_table)
            dst_node = BaseSingleQueryBaseNode(dst_conn, cols=['*'])
        elif isinstance(dst_db_table, BaseSingleQueryBaseNode):
            dst_node = dst_db_table
        else:
            raise ValueError('dst_db_table is not valid! please check!')
        
        if no_self_update and self.db_table == dst_node.db_table:
            if self.__factor_id__ == dst_node.__factor_id__:
                dst = dst_node.db_table
                src = self.db_table
                raise ValueError(
                    f'Detect self-update process! these operator attempts to update data from {src} to {dst}')

        update_status = 'full' if dst_node.empty else 'incremental'

        func = getattr(UpdateSQLUtils, f'{update_status}_update')
        sql = func(self, dst_node.db_table, fid_col=fid_ck, dt_col=dt_max_1st)
        if execute:
            self.operator(sql)
        return sql, update_status

    def save_to(self, dst_db_table):
        """
        Save current query results to target table.
        
        Convenience method using >> operator.
        
        Args:
            dst_db_table: Target table name
            
        Returns:
            Tuple of (sql, update_status)
        """
        return self >> dst_db_table


class BaseSingleFactorTableNode(BaseSingleQueryBaseNode):
    """
    Factor table node for quantitative analysis.
    
    This class extends BaseSingleQueryBaseNode with additional
    features specific to factor-based quantitative analysis.
    
    Attributes:
        _complex: Whether query is marked as complex
        _execute: Whether to auto-execute queries
        
    Example:
        >>> factor = BaseSingleFactorTableNode(
        >>>     "clickhouse://user:pass@host:8123/db.factor_table",
        >>>     cols=['*'],
        >>>     date='2024-01-01',
        >>>     limit=1000
        >>> )
        >>> df = factor.fetch()
    """
    
    __slots__ = (
        'operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', 'status', '_INFO', 'depend_tables',
        '_execute', '_no_self_update', '_complex', 'src'
    )

    def __init__(self, src: str, db_table: (None, str) = None, info=None, execute: bool = False, pool=False,
                 **kwargs):
        """
        Initialize factor table node.
        
        Args:
            src: ClickHouse connection string
            db_table: Database and table name
            info: Optional metadata
            execute: Whether to auto-execute queries
            pool: Whether to use connection pool
            **kwargs: Query parameters
        """
        super(BaseSingleFactorTableNode, self).__init__(src, db_table=db_table, info=info, pool=pool, **kwargs)

        self._complex = False
        self._execute = execute


if __name__ == '__main__':
    pass
