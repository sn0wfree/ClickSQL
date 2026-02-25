# coding=utf-8
"""
Connection Management Module.

This module provides connection pooling and session management for ClickHouse HTTP client.
It implements a thread-safe singleton session pool for connection reuse, which can
significantly improve performance for multiple queries.

Classes:
    SessionPool: Thread-safe singleton session pool for HTTP connections
    ConnectionManager: Alternative connection manager with advanced features

Functions:
    get_session_pool: Get the global session pool instance

Example:
    >>> from ClickSQL.clickhouse.connection import get_session_pool
    >>> pool = get_session_pool()
    >>> session = pool.get_session("localhost:8123:db", pool_size=5)
"""
import asyncio
import gzip
import threading
import warnings
from functools import partial
from urllib import parse
from contextlib import contextmanager

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ClickSQL.conf import Config
from ClickSQL.errors import DatabaseError, ServerError
from ClickSQL.clickhouse.format import ClickHouseHelper

ENGAGE_ASYNC = Config.get('ENGAGE_ASYNC', default=False)
if ENGAGE_ASYNC:
    try:
        import nest_asyncio
        nest_asyncio.apply()
        from aiohttp import ClientSession, TCPConnector
    except Exception:
        ENGAGE_ASYNC = False
        from requests import Session as ClientSession
else:
    from requests import Session as ClientSession

GLOBAL_RAISE_ERROR = Config.get('GLOBAL_RAISE_ERROR', default=True)
SEMAPHORE = Config.get('SEMAPHORE', default=10)

DEFAULT_SETTINGS = {
    'enable_http_compression': 1,
    'send_progress_in_http_headers': 0,
    'log_queries': 1,
    'connect_timeout': 10,
    'receive_timeout': 300,
    'send_timeout': 300,
    'output_format_json_quote_64bit_integers': 0,
    'wait_end_of_query': 0
}


class SessionPool:
    """
    Thread-safe singleton session pool for HTTP connection reuse.
    
    This class implements a singleton pattern to ensure only one session pool
    exists across the application. Each session in the pool maintains persistent
    connections to ClickHouse servers, reducing connection overhead.
    
    Attributes:
        _sessions: Dictionary of active sessions keyed by pool key
        _session_lock: Thread lock for session access
        
    Note:
        This is a singleton - use get_session_pool() to get the instance
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions = {}
                    cls._instance._session_lock = threading.Lock()
        return cls._instance
    
    def get_session(self, pool_key: str, pool_size: int = 5) -> requests.Session:
        """
        Get or create a session from the pool.
        
        Args:
            pool_key: Unique identifier for this connection (host:port:database)
            pool_size: Size of the connection pool
            
        Returns:
            requests.Session object for HTTP connections
        """
        with self._session_lock:
            if pool_key not in self._sessions:
                session = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=pool_size,
                    pool_maxsize=pool_size,
                    max_retries=Retry(total=3, backoff_factor=0.1)
                )
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                self._sessions[pool_key] = session
            return self._sessions[pool_key]
    
    def close_all(self):
        """
        Close all sessions in the pool.
        
        Releases all HTTP connections. Call this when shutting down the application.
        """
        with self._session_lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()


_session_pool = SessionPool()


def get_session_pool() -> SessionPool:
    """Get the global session pool instance."""
    return _session_pool


class ConnectionManager(ClickHouseHelper):
    """Manages HTTP connections to ClickHouse with performance optimizations."""

    __slots__ = ('_connect_url', '_para', 'http_settings', '_pool_key', '_cached_url', '_pool_size', '_reuse_session')

    def __init__(self, host: str, port: int, user: str, password: str, database: str, 
                 pool_size: int = 5, reuse_session: bool = True):
        self._para = {'host': host, 'port': port, 'user': user, 'password': password, 'database': database}
        self._connect_url = f'http://{user}:{password}@{host}:{port}'
        self._pool_key = f"{host}:{port}:{database}"
        self._cached_url = None
        self._pool_size = pool_size
        self._reuse_session = reuse_session
        self.http_settings = self._default_settings()
    
    @staticmethod
    def _default_settings() -> dict:
        """Get default HTTP settings."""
        return DEFAULT_SETTINGS.copy()
    
    @staticmethod
    def _merge_settings(user_settings: dict = None, default_settings: dict = None) -> dict:
        """Merge user settings with defaults."""
        settings = (default_settings or DEFAULT_SETTINGS).copy()
        if user_settings:
            settings.update(user_settings)
        return settings

    def _build_url(self) -> str:
        """Get cached URL for faster access."""
        if self._cached_url is None:
            self._cached_url = f'{self._connect_url}/?{parse.urlencode(self.http_settings)}'
        return self._cached_url

    @contextmanager
    def _get_session(self):
        """Get a session - either from pool or new session based on reuse_session setting."""
        if self._reuse_session:
            session = get_session_pool().get_session(self._pool_key, self._pool_size)
            yield session
        else:
            session = requests.Session()
            try:
                yield session
            finally:
                session.close()

    def _post(self, url: str, sql: str, session, raise_error: bool = True):
        """Synchronous POST request."""
        if self.http_settings['enable_http_compression'] == 1:
            with session.post(url, data=gzip.compress(sql.encode()),
                              headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'}) as resp:
                result = resp.content
        else:
            with session.post(url, body=sql.encode()) as resp:
                result = resp.content

        if resp.status_code != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise ServerError(result)
            warnings.warn(str(result))
        return result

    async def _post_async(self, url: str, sql: str, session, raise_error: bool = True):
        """Asynchronous POST request."""
        if self.http_settings['enable_http_compression'] == 1:
            async with session.post(url, data=gzip.compress(sql.encode()),
                                   headers={'Content-Encoding': 'gzip', 'Accept-Encoding': 'gzip'}) as resp:
                result = await resp.read()
        else:
            async with session.post(url, body=sql.encode()) as resp:
                result = await resp.read()

        if resp.status != 200:
            if raise_error and GLOBAL_RAISE_ERROR:
                raise DatabaseError(result)
            warnings.warn(str(result))
        return result

    def execute(self, sql, convert_to='dataframe', transfer_sql_format=True, 
                to_df=True, raise_error=True, async_mode=True):
        """Execute SQL synchronously or asynchronously."""
        if async_mode and ENGAGE_ASYNC:
            return self._execute_async(sql, convert_to, transfer_sql_format, raise_error)
        return self._execute_sync(sql, convert_to, transfer_sql_format, to_df, raise_error)

    def _execute_sync(self, sql, convert_to, transfer_sql_format, to_df, raise_error):
        """Synchronous execution with optional session reuse."""
        url = self._build_url()
        transfer_sql = partial(self._transfer_sql_format, convert_to=convert_to,
                               transfer_sql_format=transfer_sql_format)
        
        with self._get_session() as session:
            if isinstance(sql, str):
                result = self._post(url, transfer_sql(sql), session, raise_error)
            elif isinstance(sql, (list, tuple)):
                result = [self._post(url, transfer_sql(s), session, raise_error) for s in sql]
            else:
                raise ValueError('sql must be str, list or tuple')
        
        return self._parse_result(sql, result, convert_to, to_df)

    async def _execute_async(self, sql, convert_to, transfer_sql_format, raise_error):
        """Asynchronous execution with connection pool."""
        url = self._build_url()
        transfer_sql = partial(self._transfer_sql_format, convert_to=convert_to,
                               transfer_sql_format=transfer_sql_format)
        sem = asyncio.Semaphore(SEMAPHORE)
        
        async with sem:
            connector = TCPConnector(limit=SEMAPHORE, limit_per_host=SEMAPHORE)
            async with ClientSession(connector=connector) as session:
                if isinstance(sql, str):
                    result = await self._post_async(url, transfer_sql(sql), session, raise_error)
                elif isinstance(sql, (list, tuple)):
                    result = [await self._post_async(url, transfer_sql(s), session, raise_error) for s in sql]
                else:
                    raise ValueError('sql must be str, list or tuple')
        
        return self._parse_result(sql, result, convert_to, to_df=True)

    def _parse_result(self, sql, result, convert_to, to_df):
        """Parse result into DataFrame or return raw."""
        if not to_df:
            return result
        
        if isinstance(sql, str):
            return self._load_into_pd(result, convert_to) if result != b'' else result
        return [self._load_into_pd(s, convert_to) if s != b'' else s for s in result]
    
    def close(self):
        """Close the connection pool for this connection."""
        pass  # Sessions are pooled globally
    
    @classmethod
    def close_all_pools(cls):
        """Close all session pools."""
        get_session_pool().close_all()
