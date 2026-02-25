# coding=utf-8
"""
Nodes Module - Query Node Classes for ClickSQL.

This module provides node-based query abstractions for ClickHouse databases,
including factor table nodes, group by operations, and merge operations.

Classes:
    - BaseSingleQueryBaseNode: Base class for single query operations
    - BaseSingleFactorTableNode: Factor table node for quantitative analysis
    - DelayTasks: Queue for delayed SQL execution
    - GroupSQLUtils: SQL generation utilities for GROUP BY operations
    - MergeSQLUtils: SQL generation utilities for MERGE/JOIN operations

Example:
    >>> from ClickSQL.nodes import BaseSingleFactorTableNode
    >>> node = BaseSingleFactorTableNode("clickhouse://user:pass@host:8123/db.table")
    >>> result = node.fetch(100)
"""

from ClickSQL.nodes.base import BaseSingleQueryBaseNode, BaseSingleFactorTableNode, DelayTasks
from ClickSQL.nodes.groupby import GroupSQLUtils
from ClickSQL.nodes.merge import MergeSQLUtils

__all__ = [
    'BaseSingleQueryBaseNode',
    'BaseSingleFactorTableNode',
    'DelayTasks',
    'GroupSQLUtils',
    'MergeSQLUtils'
]
