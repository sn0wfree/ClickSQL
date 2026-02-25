# coding=utf-8
"""ClickSQL MCP Server."""
from ClickSQL.mcp import tools

__all__ = ['tools']
__version__ = '0.1.10'

try:
    from ClickSQL.mcp.server import mcp, run
    __all__.extend(['mcp', 'run'])
except ImportError:
    mcp = None
    run = None
