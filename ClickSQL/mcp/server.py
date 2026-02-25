# coding=utf-8
"""MCP Server for ClickSQL."""
import os

try:
    from fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    FastMCP = None

from ClickSQL.mcp import tools as _tools

# MCP 实例，仅在 fastmcp 可用时创建
mcp = None
if HAS_FASTMCP:
    mcp = FastMCP("ClickSQL")
    
    mcp.tool()(_tools.execute_query)
    mcp.tool()(_tools.get_tables)
    mcp.tool()(_tools.get_columns)
    mcp.tool()(_tools.get_databases)
    mcp.tool()(_tools.insert_dataframe)
    mcp.tool()(_tools.create_table)
    mcp.tool()(_tools.get_table_count)
    mcp.tool()(_tools.table_exists)
    mcp.tool()(_tools.execute_async_query)
    
    @mcp.tool()
    def list_connections() -> list:
        """List all active ClickHouse connections."""
        return _tools.list_connections()
    
    @mcp.prompt()
    def query_prompt() -> str:
        """Prompt template for SQL queries."""
        return """You are a ClickHouse database expert. 
    Use the execute_query tool to run SQL queries against ClickHouse.
    Always verify your SQL syntax before executing.
    Return results in a readable format."""
    
    @mcp.prompt()
    def insert_prompt() -> str:
        """Prompt template for data insertion."""
        return """You are inserting data into ClickHouse.
    Use the insert_dataframe tool with:
    - data: list of dictionaries
    - db: target database
    - table: target table
    - chunksize: rows per batch (default 100000)
    
    Ensure data types match the target table schema."""


DEFAULT_CONN_ENV = "CLICKHOUSE_CONNECTION"


def get_default_connection() -> str:
    """Get default connection from environment variable."""
    return os.environ.get(DEFAULT_CONN_ENV, "")


def check_fastmcp():
    """检查 fastmcp 是否可用"""
    if not HAS_FASTMCP:
        raise ImportError(
            "fastmcp is required to run MCP server. "
            "Please install: pip install fastmcp (requires Python 3.10+)"
        )


def run(transport: str = "stdio"):
    """
    Run the MCP server.
    
    Args:
        transport: Transport type - "stdio" or "sse"
    """
    check_fastmcp()
    
    if transport == "sse":
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8080"))
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    run(transport)
