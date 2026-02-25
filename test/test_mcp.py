# coding=utf-8
"""MCP 工具单元测试"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


class TestMCPTools(unittest.TestCase):
    """测试 ClickSQL MCP 工具函数"""
    
    @classmethod
    def setUpClass(cls):
        """测试类前清理连接缓存"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL.mcp import tools
        tools._connections.clear()
    
    def setUp(self):
        """测试前准备"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL.mcp import tools
        self.tools = tools
        
        self.conn_str = "clickhouse://default:password@localhost:8123/test_db"
        
        # 创建模拟节点
        self.mock_node = Mock()
        self.mock_node.execute.return_value = pd.DataFrame({
            'name': ['system', 'default', 'test_db']
        })
        self.mock_node.tables = ['users', 'orders', 'products']
        self.mock_node.query.return_value = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        self.mock_node.insert_df.return_value = None
        self.mock_node._check_exists.return_value = True
    
    def tearDown(self):
        """测试后清理"""
        self.tools._connections.clear()
    
    def test_get_connection(self):
        """测试获取连接"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            # 首次调用创建连接
            conn1 = self.tools.get_connection(self.conn_str)
            self.assertIsNotNone(conn1)
            
            # 再次调用返回相同连接
            conn2 = self.tools.get_connection(self.conn_str)
            self.assertIs(conn1, conn2)
            
            # 验证节点被创建
            MockNode.assert_called_once()
    
    def test_execute_query(self):
        """测试执行 SQL 查询"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            # 测试返回 DataFrame
            result = self.tools.execute_query(self.conn_str, "SELECT * FROM users")
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 3)
            
            # 测试返回字典列表
            result_dict = self.tools.execute_query(self.conn_str, "SELECT * FROM users", output_df=False)
            self.assertIsInstance(result_dict, list)
            self.assertEqual(len(result_dict), 3)
    
    def test_get_databases(self):
        """测试获取数据库列表"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            databases = self.tools.get_databases(self.conn_str)
            self.assertIsInstance(databases, list)
            self.assertIn('system', databases)
            self.assertIn('default', databases)
    
    def test_get_tables(self):
        """测试获取表列表"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            # 测试使用默认数据库
            tables = self.tools.get_tables(self.conn_str)
            self.assertEqual(tables, ['users', 'orders', 'products'])
            
            # 测试指定数据库
            self.mock_node.execute.return_value = pd.DataFrame({
                'name': ['table1', 'table2']
            })
            tables = self.tools.get_tables(self.conn_str, database='test_db')
            self.assertIsInstance(tables, list)
    
    def test_get_columns(self):
        """测试获取表结构"""
        # 模拟 DESCRIBE TABLE 结果
        describe_result = pd.DataFrame({
            'name': ['id', 'name', 'email', 'created_at'],
            'type': ['UInt32', 'String', 'String', 'DateTime'],
            'default_type': ['', '', '', ''],
            'default_expression': ['', '', '', ''],
            'comment': ['', '', '', ''],
            'codec_expression': ['', '', '', ''],
            'ttl_expression': ['', '', '', '']
        })
        self.mock_node.execute.return_value = describe_result
        
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            columns = self.tools.get_columns(self.conn_str, 'test_db', 'users')
            self.assertIsInstance(columns, list)
            self.assertEqual(len(columns), 4)
            self.assertEqual(columns[0]['name'], 'id')
    
    def test_insert_dataframe(self):
        """测试插入数据"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            test_data = [
                {'id': 100, 'name': 'Test User 1'},
                {'id': 101, 'name': 'Test User 2'}
            ]
            
            result = self.tools.insert_dataframe(self.conn_str, test_data, 'test_db', 'users')
            
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['rows_inserted'], 2)
            self.assertEqual(result['table'], 'test_db.users')
            
            # 验证 insert_df 被调用
            self.mock_node.insert_df.assert_called_once()
    
    def test_create_table(self):
        """测试创建表"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            create_sql = """
            CREATE TABLE test_db.users (
                id UInt32,
                name String
            ) ENGINE = Memory()
            """
            
            result = self.tools.create_table(self.conn_str, 'test_db', 'users', create_sql)
            
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['table'], 'test_db.users')
            
            # 验证 execute 被调用
            self.mock_node.execute.assert_called()
    
    def test_get_table_count(self):
        """测试获取表行数"""
        # 模拟 count 查询结果
        self.mock_node.execute.return_value = pd.DataFrame({'cnt': [1000]})
        
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            count = self.tools.get_table_count(self.conn_str, 'test_db', 'users')
            
            self.assertEqual(count, 1000)
    
    def test_table_exists(self):
        """测试表存在性检查"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            # 表存在
            exists = self.tools.table_exists(self.conn_str, 'test_db', 'users')
            self.assertTrue(exists)
            
            # 表不存在
            self.mock_node._check_exists.return_value = False
            exists = self.tools.table_exists(self.conn_str, 'test_db', 'nonexistent')
            self.assertFalse(exists)
    
    def test_execute_async_query(self):
        """测试异步查询"""
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            result = self.tools.execute_async_query(self.conn_str, "SELECT * FROM users")
            
            # 验证 query 被调用
            self.mock_node.query.assert_called()
    
    def test_list_connections(self):
        """测试列出连接"""
        # 清空连接
        self.tools._connections.clear()
        
        with patch.object(self.tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = self.mock_node
            
            # 创建连接
            self.tools.get_connection(self.conn_str)
            
            # 列出连接
            connections = self.tools.list_connections()
            self.assertIn(self.conn_str, connections)
    
    def test_set_connection(self):
        """测试手动设置连接"""
        # 清空连接
        self.tools._connections.clear()
        
        custom_node = Mock()
        self.tools.set_connection(self.conn_str, custom_node)
        
        # 验证连接已设置
        conn = self.tools.get_connection(self.conn_str)
        self.assertIs(conn, custom_node)


class TestMCPToolsEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    @classmethod
    def setUpClass(cls):
        """测试类前清理连接缓存"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL.mcp import tools
        tools._connections.clear()
    
    def setUp(self):
        """测试前准备"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL.mcp import tools
        tools._connections.clear()
        self.tools = tools
    
    def tearDown(self):
        """测试后清理"""
        self.tools._connections.clear()
    
    def test_insert_dataframe_requires_pandas(self):
        """测试插入需要 pandas"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from ClickSQL.mcp import tools
        
        # 模拟没有 pandas 的情况
        with patch.object(tools, 'HAS_PANDAS', False):
            with self.assertRaises(ImportError):
                tools.insert_dataframe(
                    "clickhouse://localhost:8123/test",
                    [{'id': 1}],
                    'test_db',
                    'users'
                )
    
    def test_execute_query_empty_result(self):
        """测试空结果"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        mock_node = Mock()
        mock_node.query.return_value = pd.DataFrame()
        
        from ClickSQL.mcp import tools
        
        with patch.object(tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = mock_node
            
            result = tools.execute_query("clickhouse://localhost:8123/test", "SELECT * FROM empty")
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 0)


class TestMCPToolsIntegration(unittest.TestCase):
    """集成测试 - 模拟完整工作流"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 创建模拟节点
        mock_node = Mock()
        
        # 数据库列表
        mock_node.execute.return_value = pd.DataFrame({
            'name': ['system', 'default', 'test_db']
        })
        mock_node.tables = ['users', 'orders']
        
        # 查询结果
        mock_node.query.return_value = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        mock_node.insert_df.return_value = None
        mock_node._check_exists.return_value = True
        
        conn_str = "clickhouse://default:password@localhost:8123/test_db"
        
        from ClickSQL.mcp import tools
        
        with patch.object(tools, 'ClickHouseTableNodeExt') as MockNode:
            MockNode.return_value = mock_node
            
            # 1. 获取数据库
            dbs = tools.get_databases(conn_str)
            self.assertIn('test_db', dbs)
            
            # 2. 获取表
            tables = tools.get_tables(conn_str)
            self.assertIn('users', tables)
            
            # 3. 查询数据
            result = tools.execute_query(conn_str, "SELECT * FROM users")
            self.assertEqual(len(result), 3)
            
            # 4. 插入数据
            test_data = [{'id': 10, 'name': 'Test'}]
            insert_result = tools.insert_dataframe(conn_str, test_data, 'test_db', 'users')
            self.assertEqual(insert_result['status'], 'success')
            
            # 5. 检查表存在
            exists = tools.table_exists(conn_str, 'test_db', 'users')
            self.assertTrue(exists)


if __name__ == '__main__':
    unittest.main()
