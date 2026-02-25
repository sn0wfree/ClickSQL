# coding=utf-8
"""
ClickSQL MCP 使用案例

这个脚本展示了如何使用 ClickSQL 的 MCP 工具包。
包含两种使用方式：
1. 直接调用工具函数
2. MCP 服务器方式 (需要配置 Claude Desktop 等客户端)

运行方式:
    python examples/mcp_example.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# ========== 模拟设置 ==========
def mock_clickhouse_node():
    """创建模拟的 ClickHouse 节点"""
    mock_node = Mock()
    
    # 模拟数据库列表
    mock_node.execute.return_value = pd.DataFrame({
        'name': ['system', 'default', 'test_db']
    })
    
    # 模拟表列表
    mock_node.tables = ['users', 'orders', 'products']
    
    # 模拟查询结果
    mock_node.query.return_value = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })
    
    return mock_node


def mock_clickhouse_node_ext():
    """创建模拟的 ClickHouseTableNodeExt 节点"""
    mock_node = Mock()
    
    # 模拟数据库
    mock_node.execute.return_value = pd.DataFrame({
        'name': ['system', 'default', 'test_db']
    })
    
    # 模拟表列表
    mock_node.tables = ['users', 'orders', 'products', 'logs']
    
    # 模拟查询结果 - users 表
    mock_node.query.return_value = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                  'david@example.com', 'eve@example.com'],
        'created_at': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', 
                                       '2024-01-04', '2024-01-05'])
    })
    
    # 模拟 insert_df
    mock_node.insert_df.return_value = None
    
    # 模拟 _check_exists
    mock_node._check_exists.return_value = True
    
    return mock_node


# ========== 演示代码 ==========
def demo_direct_tools():
    """演示直接调用工具函数"""
    print("=" * 60)
    print("演示 1: 直接调用 MCP 工具函数")
    print("=" * 60)
    
    # 导入工具模块
    from ClickSQL.mcp import tools
    
    # 使用 mock 模拟 ClickHouseTableNodeExt
    with patch.object(tools, 'ClickHouseTableNodeExt') as MockNode:
        mock_node = mock_clickhouse_node_ext()
        MockNode.return_value = mock_node
        
        # 导入工具模块
        from ClickSQL.mcp import tools
        
        conn_str = "clickhouse://default:password@localhost:8123/test_db"
        
        # 1. 获取数据库列表
        print("\n[1] 获取数据库列表:")
        databases = tools.get_databases(conn_str)
        print(f"    数据库: {databases}")
        
        # 2. 获取表列表
        print("\n[2] 获取表列表:")
        tables = tools.get_tables(conn_str)
        print(f"    表: {tables}")
        
        # 3. 执行查询
        print("\n[3] 执行 SQL 查询:")
        result = tools.execute_query(conn_str, "SELECT * FROM users LIMIT 3")
        print(f"    结果:\n{result}")
        
        # 4. 获取表结构
        print("\n[4] 获取表结构:")
        with patch.object(mock_node, 'execute') as mock_execute:
            mock_execute.return_value = pd.DataFrame({
                'name': ['id', 'name', 'email', 'created_at'],
                'type': ['UInt32', 'String', 'String', 'DateTime'],
                'default_type': ['', '', '', '']
            })
            columns = tools.get_columns(conn_str, 'test_db', 'users')
            print(f"    列信息: {columns[:2]}...")
        
        # 5. 插入数据
        print("\n[5] 插入数据:")
        test_data = [
            {'id': 100, 'name': 'Test User', 'email': 'test@example.com'},
            {'id': 101, 'name': 'Another User', 'email': 'another@example.com'}
        ]
        result = tools.insert_dataframe(conn_str, test_data, 'test_db', 'users')
        print(f"    插入结果: {result}")
        
        # 6. 检查表是否存在
        print("\n[6] 检查表是否存在:")
        exists = tools.table_exists(conn_str, 'test_db', 'users')
        print(f"    users 表存在: {exists}")
        
        # 7. 获取表行数
        print("\n[7] 获取表行数:")
        with patch.object(mock_node, 'execute') as mock_execute:
            mock_execute.return_value = pd.DataFrame({'cnt': [1000]})
            count = tools.get_table_count(conn_str, 'test_db', 'users')
            print(f"    users 表行数: {count}")


def demo_mcp_server():
    """演示 MCP 服务器模式"""
    print("\n" + "=" * 60)
    print("演示 2: MCP 服务器模式")
    print("=" * 60)
    
    print("""
要使用 MCP 服务器模式，需要:

1. 安装依赖:
   pip install fastmcp

2. 配置 Claude Desktop (Windows):
   复制 examples/mcp_config.json 到 %USERPROFILE%\\.claude.json
   
3. 修改连接字符串为您的 ClickHouse 地址

4. 重启 Claude Desktop

5. 使用自然语言查询，例如:
   - "查询 users 表的前 10 条数据"
   - "列出所有数据库"
   - "查看 orders 表的结构"
   - "插入一条新数据到 test 表"
""")


def demo_async_operations():
    """演示异步操作"""
    print("\n" + "=" * 60)
    print("演示 3: 异步操作")
    print("=" * 60)
    
    from ClickSQL.mcp import tools
    
    with patch.object(tools, 'ClickHouseTableNodeExt') as MockNode:
        mock_node = mock_clickhouse_node_ext()
        MockNode.return_value = mock_node
        
        conn_str = "clickhouse://default:password@localhost:8123/test_db"
        
        # 模拟异步查询
        print("\n执行异步查询...")
        result = tools.execute_async_query(conn_str, "SELECT * FROM users", async_mode=True)
        print(f"    查询完成，结果类型: {type(result)}")
        
        # 注意: 由于 async_mode 参数传递方式，mock 可能不会触发


def demo_mcp_tools_definition():
    """展示 MCP 工具定义"""
    print("\n" + "=" * 60)
    print("MCP 工具列表")
    print("=" * 60)
    
    from ClickSQL.mcp import tools
    
    tool_list = [
        ("execute_query", "执行 SQL 查询"),
        ("get_tables", "获取表列表"),
        ("get_columns", "获取表结构"),
        ("get_databases", "获取数据库列表"),
        ("insert_dataframe", "插入数据"),
        ("create_table", "创建表"),
        ("get_table_count", "获取行数"),
        ("table_exists", "检查表是否存在"),
        ("execute_async_query", "异步执行查询"),
    ]
    
    print("\n可用工具:")
    for name, desc in tool_list:
        print(f"  - {name}: {desc}")


def demo_dataframe_conversion():
    """演示 DataFrame 转换"""
    print("\n" + "=" * 60)
    print("演示 4: DataFrame 转换")
    print("=" * 60)
    
    # 创建测试 DataFrame
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'score': [95.5, 87.3, 92.1]
    })
    
    print("\n原始 DataFrame:")
    print(df)
    
    # 转换为字典列表
    records = df.to_dict('records')
    print("\n转换为字典列表:")
    print(json.dumps(records, indent=2, default=str))


# ========== 主函数 ==========
def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════╗
║           ClickSQL MCP 使用案例                              ║
║           Model Context Protocol for ClickHouse             ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # 演示各种功能
    demo_mcp_tools_definition()
    demo_direct_tools()
    demo_async_operations()
    demo_dataframe_conversion()
    demo_mcp_server()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
