# -*- coding: utf-8 -*-
"""
ClickSQL 完整使用指南
==================

本文件提供 ClickSQL 的详细使用示例，包括：
1. 基本连接和查询
2. 数据插入
3. MCP 工具使用
4. 性能优化
5. 异步操作
"""

# ============================================================================
# 第一部分：基本使用
# ============================================================================

def basic_usage():
    """基本使用示例"""
    
    # 导入
    from ClickSQL import ClickHouseTableNode, ClickHouseTableNodeExt
    
    # 连接字符串格式
    # clickhouse://username:password@host:port/database
    
    # 1. 简单连接
    conn_str = "clickhouse://default:password@localhost:8123/my_database"
    node = ClickHouseTableNode(conn_str)
    
    # 2. 查询数据
    result = node.query("SELECT * FROM users LIMIT 10")
    print(result.head())
    
    # 3. 执行 SQL
    result = node.execute("SELECT count(*) as cnt FROM orders")
    print(result)
    
    # 4. 获取表列表
    print(node.tables)
    
    # 5. 获取数据库列表
    print(node.databases)


# ============================================================================
# 第二部分：数据插入
# ============================================================================

def insert_usage():
    """数据插入示例"""
    
    import pandas as pd
    from ClickSQL import ClickHouseTableNode
    
    conn_str = "clickhouse://default:password@localhost:8123/my_database"
    node = ClickHouseTableNode(conn_str)
    
    # 方法1: 从 DataFrame 插入
    df = pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'User_{i}' for i in range(1, 1001)],
        'score': [round(60 + i * 0.04, 2) for i in range(1000)]
    })
    
    # 同步插入
    node.insert_df(df, 'my_database', 'users')
    
    # 方法2: 大数据分块插入
    large_df = pd.DataFrame({
        'id': range(1, 100001),
        'data': ['x' * 100] * 100000
    })
    node.insert_df(large_df, 'my_database', 'logs', chunksize=50000)
    
    # 方法3: 并行插入 (多线程)
    node.insert_df(large_df, 'my_database', 'logs', 
                   parallel=True, max_workers=8)


# ============================================================================
# 第三部分：MCP 工具
# ============================================================================

def mcp_tools_usage():
    """MCP 工具使用示例"""
    
    from ClickSQL.mcp import tools
    
    conn_str = "clickhouse://default:password@localhost:8123/my_db"
    
    # 1. 执行查询
    result = tools.execute_query(conn_str, "SELECT * FROM users LIMIT 10")
    print(result)
    
    # 2. 获取数据库列表
    databases = tools.get_databases(conn_str)
    print(databases)
    
    # 3. 获取表列表
    tables = tools.get_tables(conn_str)
    print(tables)
    
    # 4. 获取表结构
    columns = tools.get_columns(conn_str, 'my_db', 'users')
    for col in columns:
        print(f"{col['name']}: {col['type']}")
    
    # 5. 插入数据
    data = [
        {'id': 1, 'name': 'Alice', 'score': 95.5},
        {'id': 2, 'name': 'Bob', 'score': 87.3},
    ]
    result = tools.insert_dataframe(conn_str, data, 'my_db', 'users')
    print(result)  # {'status': 'success', 'rows_inserted': 2}
    
    # 6. 检查表是否存在
    exists = tools.table_exists(conn_str, 'my_db', 'users')
    print(exists)  # True
    
    # 7. 获取表行数
    count = tools.get_table_count(conn_str, 'my_db', 'users')
    print(count)
    
    # 8. 创建表
    create_sql = """
    CREATE TABLE my_db.new_table (
        id UInt32,
        name String,
        created_at DateTime
    ) ENGINE = Memory()
    """
    result = tools.create_table(conn_str, 'my_db', 'new_table', create_sql)
    print(result)


# ============================================================================
# 第四部分：异步操作
# ============================================================================

def async_usage():
    """异步操作示例"""
    
    import asyncio
    import pandas as pd
    from ClickSQL import ClickHouseTableNode
    
    async def main():
        conn_str = "clickhouse://default:password@localhost:8123/my_db"
        node = ClickHouseTableNode(conn_str)
        
        # 异步查询
        result = node.query("SELECT * FROM users", async_mode=True)
        
        # 异步插入 DataFrame
        df = pd.DataFrame({
            'id': range(1, 1001),
            'data': ['test'] * 1000
        })
        await node.insert_df_async(df, 'my_db', 'logs', max_concurrent=4)
        
        # 批量异步查询
        queries = [
            "SELECT count(*) FROM users",
            "SELECT count(*) FROM orders", 
            "SELECT count(*) FROM products"
        ]
        results = node.execute(*queries, async_mode=True)
        for r in results:
            print(r)
    
    asyncio.run(main())


# ============================================================================
# 第五部分：性能优化
# ============================================================================

def performance_usage():
    """性能优化示例"""
    
    from ClickSQL import ClickHouseTableNode
    from ClickSQL.utils import get_query_cache, clear_query_cache
    
    conn_str = "clickhouse://default:password@localhost:8123/my_db"
    node = ClickHouseTableNode(conn_str)
    
    # 1. 启用内存缓存
    result = node.execute(
        "SELECT * FROM config_table",
        enable_memory_cache=True,
        # 可选: 自定义 TTL (秒)
        # cache_ttl=300
    )
    
    # 2. 手动管理缓存
    cache = get_query_cache()
    print(f"缓存条目: {len(cache)}")
    cache.clear()
    
    # 3. 使用文件缓存
    result = node.execute(
        "SELECT * FROM static_data",
        enable_cache=True,
        exploit_func=True
    )
    
    # 4. 批量插入优化
    import pandas as pd
    df = pd.DataFrame({
        'id': range(1, 100001),
        'value': range(100001)
    })
    
    # 大块减少网络开销，小块提高内存使用
    node.insert_df(df, 'my_db', 'large_table', chunksize=100000)
    
    # 5. 并行插入
    node.insert_df(df, 'my_db', 'large_table', 
                   parallel=True, max_workers=8)


# ============================================================================
# 第六部分：连接池
# ============================================================================

def pool_usage():
    """连接池使用示例"""
    
    from ClickSQL.pool import PooledClickHouseTableNodeExt
    
    # 创建带连接池的节点
    pool_node = PooledClickHouseTableNodeExt(
        "clickhouse://default:password@localhost:8123/my_db",
        num_pools=10  # 连接池数量
    )
    
    # 使用方式相同
    result = pool_node.query("SELECT * FROM users LIMIT 10")
    print(result)
    
    # 关闭连接池
    pool_node.close()


# ============================================================================
# 第七部分：高级查询
# ============================================================================

def advanced_query_usage():
    """高级查询示例"""
    
    from ClickSQL import ClickHouseTableNodeExt
    
    conn_str = "clickhouse://default:password@localhost:8123/my_db"
    node = ClickHouseTableNodeExt(conn_str)
    
    # 1. 构建带条件的 SQL
    sql = node.get_sql(
        'orders',
        cols=['order_id', 'customer_id', 'total'],
        data_filter={'status': 'completed'},
        order_by_cols=['total desc'],
        limit=100
    )
    print(sql)
    
    # 2. 执行生成的 SQL
    result = node.query(sql)
    print(result)
    
    # 3. 使用缓存的表结构 (避免重复 DESCRIBE)
    sql = node.get_sql('users', use_cache=True)
    
    # 4. 清理表结构缓存
    node.clear_table_cache('users')  # 清理特定表
    node.clear_table_cache()  # 清理所有


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("ClickSQL 使用指南")
    print("=" * 50)
    print("请查看函数注释了解各部分用法")
    print()
    print("基本使用: basic_usage()")
    print("数据插入: insert_usage()")
    print("MCP工具: mcp_tools_usage()")
    print("异步操作: async_usage()")
    print("性能优化: performance_usage()")
    print("连接池: pool_usage()")
    print("高级查询: advanced_query_usage()")
