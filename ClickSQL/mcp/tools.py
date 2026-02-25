# coding=utf-8
"""
MCP (Model Context Protocol) Tools for ClickSQL.

This module provides a collection of tool functions for AI agents to interact
with ClickHouse databases. These tools can be used with Claude Desktop,
VS Code Cline, or any other MCP-compatible AI agent.

Functions:
    - execute_query: Execute SQL query
    - get_tables: List tables in database
    - get_columns: Get table column information
    - get_databases: List databases
    - insert_dataframe: Insert data into table
    - create_table: Create a new table
    - get_table_count: Get row count
    - table_exists: Check if table exists
    - execute_async_query: Execute query asynchronously

Example:
    >>> from ClickSQL.mcp import tools
    >>> tools.execute_query("clickhouse://user:pass@host:8123/db", "SELECT 1")
"""
from typing import Any

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ClickSQL.clickhouse.ClickHouseExt import ClickHouseTableNodeExt
from ClickSQL.clickhouse.ClickHouse import ClickHouseTableNode


_connections: dict[str, ClickHouseTableNodeExt] = {}


def get_connection(conn_str: str, pool: bool = False) -> ClickHouseTableNodeExt:
    """Get or create a ClickHouse connection."""
    if conn_str not in _connections:
        if pool:
            from ClickSQL.pool.pooled_ch import PooledClickHouseTableNodeExt
            _connections[conn_str] = PooledClickHouseTableNodeExt(conn_str)
        else:
            _connections[conn_str] = ClickHouseTableNodeExt(conn_str)
    return _connections[conn_str]


def set_connection(conn_str: str, node: ClickHouseTableNodeExt):
    """Set a ClickHouse connection."""
    _connections[conn_str] = node


def list_connections() -> list[str]:
    """List all active connections."""
    return list(_connections.keys())


def execute_query(
    conn_str: str,
    sql: str,
    output_df: bool = True
) -> Any:
    """
    Execute a SQL query against ClickHouse.
    
    Args:
        conn_str: ClickHouse connection string (e.g., clickhouse://user:pass@host:port/db)
        sql: SQL query to execute
        output_df: Return as DataFrame (True) or raw dict (False)
    
    Returns:
        Query results as DataFrame or list of dicts
    """
    conn = get_connection(conn_str)
    result = conn.query(sql)
    
    if output_df and HAS_PANDAS:
        return result
    return result.to_dict('records') if hasattr(result, 'to_dict') else result


def get_tables(conn_str: str, database: str = None) -> list[str]:
    """
    Get list of tables in a database.
    
    Args:
        conn_str: ClickHouse connection string
        database: Database name (optional, uses connection default if not provided)
    
    Returns:
        List of table names
    """
    conn = get_connection(conn_str)
    if database:
        return conn.execute(f"SHOW TABLES FROM {database}", convert_to='dataframe')['name'].tolist()
    return conn.tables


def get_columns(conn_str: str, database: str, table: str) -> list[dict]:
    """
    Get column information for a table.
    
    Args:
        conn_str: ClickHouse connection string
        database: Database name
        table: Table name
    
    Returns:
        List of column info dicts
    """
    conn = get_connection(conn_str)
    result = conn.execute(f"DESCRIBE TABLE {database}.{table}", convert_to='dataframe')
    return result.to_dict('records')


def get_databases(conn_str: str) -> list[str]:
    """
    Get list of databases.
    
    Args:
        conn_str: ClickHouse connection string
    
    Returns:
        List of database names
    """
    conn = get_connection(conn_str)
    return conn.execute("SHOW DATABASES", convert_to='dataframe')['name'].tolist()


def insert_dataframe(
    conn_str: str,
    data: list[dict],
    db: str,
    table: str,
    chunksize: int = 100000
) -> dict:
    """
    Insert data into a ClickHouse table.
    
    Args:
        conn_str: ClickHouse connection string
        data: List of dictionaries to insert
        db: Target database
        table: Target table
        chunksize: Rows per chunk for insertion
    
    Returns:
        Result dict with status and row count
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for insert_dataframe")
    
    conn = get_connection(conn_str)
    df = pd.DataFrame(data)
    conn.insert_df(df, db, table, chunksize=chunksize)
    return {"status": "success", "rows_inserted": len(data), "table": f"{db}.{table}"}


def create_table(
    conn_str: str,
    db: str,
    table: str,
    sql: str
) -> dict:
    """
    Create a table using SQL.
    
    Args:
        conn_str: ClickHouse connection string
        db: Database name
        table: Table name
        sql: CREATE TABLE SQL statement
    
    Returns:
        Result dict with status
    """
    conn = get_connection(conn_str)
    conn.execute(sql)
    return {"status": "success", "table": f"{db}.{table}"}


def get_table_count(conn_str: str, database: str, table: str) -> int:
    """
    Get row count for a table.
    
    Args:
        conn_str: ClickHouse connection string
        database: Database name
        table: Table name
    
    Returns:
        Row count
    """
    conn = get_connection(conn_str)
    result = conn.execute(f"SELECT count() as cnt FROM {database}.{table}", convert_to='dataframe')
    return int(result['cnt'].iloc[0])


def table_exists(conn_str: str, database: str, table: str) -> bool:
    """
    Check if a table exists.
    
    Args:
        conn_str: ClickHouse connection string
        database: Database name
        table: Table name
    
    Returns:
        True if table exists
    """
    conn = get_connection(conn_str)
    return conn._check_exists(f"{database}.{table}", mode='table', output=False)


def execute_async_query(
    conn_str: str,
    sql: str,
    async_mode: bool = True
) -> Any:
    """
    Execute a SQL query asynchronously.
    
    Args:
        conn_str: ClickHouse connection string
        sql: SQL query to execute
        async_mode: Use async execution
    
    Returns:
        Query results as DataFrame
    """
    conn = get_connection(conn_str)
    result = conn.query(sql, async_mode=async_mode)
    return result


__all__ = [
    'get_connection',
    'set_connection',
    'list_connections',
    'execute_query',
    'get_tables',
    'get_columns',
    'get_databases',
    'insert_dataframe',
    'create_table',
    'get_table_count',
    'table_exists',
    'execute_async_query',
]
