# coding=utf-8
"""
SQL Formatting and JSON Handling Utilities.

This module provides utilities for:
- JSON serialization/deserialization with orjson support
- DataFrame to ClickHouse format conversion
- SQL type detection and formatting
- Variant type parsing

The module automatically uses orjson for 5-10x faster JSON processing if available.

Example:
    >>> from ClickSQL.clickhouse.format import ClickHouseHelper
    >>> helper = ClickHouseHelper()
    >>> lines = list(helper._check_df_and_dump(df, describe_table))
"""
import gzip
from functools import partial

import numpy as np
import pandas as pd

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json as _json
    HAS_ORJSON = False

available_queries_select = ('select', 'show', 'desc')
available_queries_insert = ('insert', 'optimize', 'create')


def _json_loads(data):
    """Unified JSON loads with orjson fallback."""
    return orjson.loads(data) if HAS_ORJSON else _json.loads(data, strict=False)


def _json_dumps(data, ensure_ascii=False):
    """Unified JSON dumps with orjson fallback."""
    if HAS_ORJSON:
        return orjson.dumps(data, ensure_ascii=ensure_ascii).decode('utf-8')
    return _json.dumps(data, ensure_ascii=ensure_ascii)


def _batch_json_dumps(rows, ensure_ascii=False):
    """Batch serialize multiple rows to JSON lines."""
    if HAS_ORJSON:
        lines = [orjson.dumps(row, option=orjson.OPT_SERIALIZE_NUMPY, ensure_ascii=ensure_ascii).decode('utf-8') for row in rows]
    else:
        lines = [_json.dumps(row, ensure_ascii=ensure_ascii) for row in rows]
    return lines


class ClickHouseHelper:
    """Helper class for SQL formatting and data conversion."""

    @staticmethod
    def _check_df_and_dump(df: pd.DataFrame, describe_table: pd.DataFrame, auto_convert=True):
        """Convert DataFrame to JSON lines for ClickHouse insertion. Optimized version."""
        describe_table = describe_table[~describe_table['default_type'].isin(['MATERIALIZED', 'ALIAS'])]
        non_nullable = describe_table[~describe_table['type'].str.startswith('Nullable')]['name'].tolist()
        integer_cols = set(describe_table[describe_table['type'].str.contains('Int', regex=False)]['name'].tolist())

        if auto_convert:
            df = df.convert_dtypes()

        missing = {i: np.where(df[i].isnull(), 1, 0).sum() for i in non_nullable}
        for col, val in missing.items():
            if val > 0:
                raise ValueError(f'"{col}" is not nullable, missing values not allowed.')

        rows = df.to_dict('records')
        col_types = {row['name']: row['type'] for _, row in describe_table.iterrows()}
        
        processed_rows = []
        for row in rows:
            for col in row:
                if pd.isnull(row[col]):
                    row[col] = None
                elif col in integer_cols:
                    try:
                        row[col] = int(row[col])
                    except (ValueError, TypeError):
                        raise ValueError(f'Column "{col}" is {col_types[col]}, value "{row[col]}" cannot be converted to Integer.')
            processed_rows.append(row)

        for line in _batch_json_dumps(processed_rows, ensure_ascii=False):
            yield line

    @staticmethod
    def _check_sql_type(sql: str) -> str:
        """Determine SQL type (select-like or insert-like)."""
        sql_lower = sql.lower()
        if sql_lower.startswith(available_queries_select):
            return 'select-liked'
        elif sql_lower.startswith(available_queries_insert):
            return 'insert-liked'
        raise ValueError('SQL type not supported')

    @staticmethod
    def _transfer_sql_format(sql: str, convert_to: str, transfer_sql_format: bool = True) -> str:
        """Add ClickHouse format clause to SQL."""
        if not transfer_sql_format:
            return sql
        
        fmt = 'JSONCompact' if convert_to.lower() == 'dataframe' else (convert_to or 'JSON')
        return sql.rstrip('; \n\t') + f' format {fmt}'

    @staticmethod
    def _load_into_pd(ret_value, convert_to: str = 'dataframe', errors='ignore'):
        """Parse ClickHouse JSON response into DataFrame. Optimized version with Variant support."""
        if convert_to.lower() != 'dataframe':
            return ret_value

        result_dict = _json_loads(ret_value)
        data = result_dict.get('data', [])
        
        if not data:
            return pd.DataFrame()

        meta = result_dict['meta']
        columns = [x['name'] for x in meta]
        
        df = pd.DataFrame(data, columns=columns)
        
        datetime_cols = [col_info['name'] for col_info in meta if col_info['type'] in ('DateTime', 'Nullable(DateTime)')]
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors=errors)
        
        variant_cols = [col_info['name'] for col_info in meta if col_info['type'].startswith('Variant')]
        for col in variant_cols:
            df[col] = df[col].apply(ClickHouseHelper._parse_variant_type)
        
        return df
    
    @staticmethod
    def _parse_variant_type(value):
        """Parse ClickHouse Variant type from JSON representation.
        
        Variant types are represented as {"TypeName": value} in JSON.
        This extracts the value from the wrapper.
        """
        if value is None:
            return None
        if isinstance(value, dict):
            if len(value) == 1:
                return list(value.values())[0]
        return value
