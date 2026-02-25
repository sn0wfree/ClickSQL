# coding=utf-8
"""Tests for new features: settings, async_insert, optimize, projections."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


class TestSettingsParameter(unittest.TestCase):
    """Test settings parameter support."""

    def test_execute_with_settings(self):
        """Test execute with settings parameter."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNode
        
        node = ClickHouseTableNode.__new__(ClickHouseTableNode)
        node._connect_url = "http://localhost:8123"
        node.http_settings = {'enable_http_compression': 1}
        node._reuse_session = False
        
        with patch.object(node, '_compression_switched_request') as mock_request:
            mock_request.return_value = b'{"data":[],"meta":[]}'
            
            settings = {'max_threads': 8, 'async_insert': 1}
            node.execute("SELECT 1", settings=settings)
            
            call_kwargs = mock_request.call_args
            self.assertEqual(call_kwargs.kwargs.get('settings'), settings)

    def test_query_with_settings(self):
        """Test query with settings parameter."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNode
        
        node = ClickHouseTableNode.__new__(ClickHouseTableNode)
        node._connect_url = "http://localhost:8123"
        node.http_settings = {'enable_http_compression': 1}
        node._reuse_session = False
        
        with patch.object(node, '_compression_switched_request') as mock_request:
            mock_request.return_value = b'{"data":[],"meta":[]}'
            
            settings = {'max_execution_time': 300}
            node.query("SELECT * FROM test", settings=settings)
            
            call_kwargs = mock_request.call_args
            self.assertEqual(call_kwargs.kwargs.get('settings'), settings)


class TestAsyncInsert(unittest.TestCase):
    """Test async_insert server-side feature."""

    def test_insert_df_async_server(self):
        """Test insert_df_async_server method."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNode
        
        node = ClickHouseTableNode.__new__(ClickHouseTableNode)
        node._connect_url = "http://localhost:8123"
        node.http_settings = {'enable_http_compression': 1}
        node._reuse_session = False
        node._db = 'db'
        
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
        
        with patch.object(node, '_prepare_insert_data') as mock_prepare:
            mock_prepare.return_value = ['INSERT INTO test VALUES ...']
            
            with patch.object(node, '_compression_switched_request') as mock_request:
                mock_request.return_value = b'{"data":[],"meta":[]}'
                
                node.insert_df_async_server(df, 'db', 'test')
                
                mock_request.assert_called()
                call_kwargs = mock_request.call_args
                settings = call_kwargs.kwargs.get('settings') if call_kwargs.kwargs else {}
                self.assertEqual(settings.get('async_insert'), 1)


class TestOptimizeAPI(unittest.TestCase):
    """Test optimize and projections API."""

    def test_optimize_table(self):
        """Test optimize_table method."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNodeExt
        
        node = ClickHouseTableNodeExt.__new__(ClickHouseTableNodeExt)
        
        with patch.object(node, 'query') as mock_query:
            mock_query.return_value = pd.DataFrame()
            
            node.optimize_table('db', 'table', final=True, deduplicate=True)
            
            mock_query.assert_called_once()
            call_sql = mock_query.call_args[0][0]
            self.assertIn('OPTIMIZE TABLE', call_sql)
            self.assertIn('FINAL', call_sql)
            self.assertIn('DEDUPLICATE', call_sql)

    def test_list_projections(self):
        """Test list_projections method."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNodeExt
        
        node = ClickHouseTableNodeExt.__new__(ClickHouseTableNodeExt)
        
        with patch.object(node, 'query') as mock_query:
            mock_query.return_value = pd.DataFrame({'name': ['proj1']})
            
            result = node.list_projections('db', 'table')
            
            self.assertIsInstance(result, pd.DataFrame)
            mock_query.assert_called_once()

    def test_system_logs(self):
        """Test system_logs method."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNodeExt
        
        node = ClickHouseTableNodeExt.__new__(ClickHouseTableNodeExt)
        
        with patch.object(node, 'query') as mock_query:
            mock_query.return_value = pd.DataFrame({'query': ['SELECT 1']})
            
            result = node.system_logs('query_log', limit=10)
            
            self.assertIsInstance(result, pd.DataFrame)
            mock_query.assert_called_once()


class TestVariantType(unittest.TestCase):
    """Test Variant type parsing."""

    def test_parse_variant_type(self):
        """Test _parse_variant_type method."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL.clickhouse.format import ClickHouseHelper
        
        result = ClickHouseHelper._parse_variant_type({"String": "hello"})
        self.assertEqual(result, "hello")
        
        result = ClickHouseHelper._parse_variant_type({"Int": 42})
        self.assertEqual(result, 42)
        
        result = ClickHouseHelper._parse_variant_type(None)
        self.assertIsNone(result)
        
        result = ClickHouseHelper._parse_variant_type("plain string")
        self.assertEqual(result, "plain string")


class TestCacheStats(unittest.TestCase):
    """Test cache statistics."""

    def test_cache_stats(self):
        """Test cache stats method."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL.utils import get_query_cache
        
        cache = get_query_cache()
        cache.clear()
        
        cache.set("SELECT 1", {"result": "test"})
        cache.get("SELECT 1")
        cache.get("SELECT 2")
        
        stats = cache.stats()
        
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertIn('hit_rate', stats)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)


class TestConnectionPool(unittest.TestCase):
    """Test connection pool functionality."""

    def test_reuse_session_parameter(self):
        """Test reuse_session parameter."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNode
        
        with patch('ClickSQL.clickhouse.ClickHouse.ClickHouseBaseNode.__init__', return_value=None):
            with patch('ClickSQL.clickhouse.ClickHouse.ClickHouseBaseNode._heartbeat_test_'):
                node = ClickHouseTableNode(host='localhost', port=8123, 
                                          user='default', password='', database='test',
                                          reuse_session=True, pool_size=5)
                
                self.assertTrue(hasattr(node, '_reuse_session'))
                self.assertTrue(hasattr(node, '_pool_size'))
                self.assertTrue(hasattr(node, '_session_pool'))
                self.assertTrue(node._reuse_session)
                self.assertEqual(node._pool_size, 5)

    def test_session_pool_disabled(self):
        """Test with session pool disabled."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL import ClickHouseTableNode
        
        with patch('ClickSQL.clickhouse.ClickHouse.ClickHouseBaseNode.__init__', return_value=None):
            with patch('ClickSQL.clickhouse.ClickHouse.ClickHouseBaseNode._heartbeat_test_'):
                node = ClickHouseTableNode(host='localhost', port=8123, 
                                          user='default', password='', database='test',
                                          reuse_session=False)
                
                self.assertFalse(node._reuse_session)


if __name__ == '__main__':
    unittest.main()
