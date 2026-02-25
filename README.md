# ClickSQL: ClickHouse client for Humans 

[![CodeQL](https://github.com/sn0wfree/ClickSQL/actions/workflows/codeql-analysis.yml/badge.svg?branch=master)](https://github.com/sn0wfree/ClickSQL/actions/workflows/codeql-analysis.yml)
[![Python package](https://github.com/sn0wfree/ClickSQL/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/sn0wfree/ClickSQL/actions/workflows/python-package.yml)

## 简介

ClickSQL 是一个 Python ClickHouse 客户端库，支持同步/异步操作，集成 MCP (Model Context Protocol)，可以方便地为 AI Agent 提供数据库工具服务。

### 主要特性

- ✅ 简洁易用的 Python API
- ✅ 支持同步/异步执行
- ✅ MCP 工具支持 (可用于 Claude Desktop、VS Code 等)
- ✅ 高性能 JSON 解析 (orjson)
- ✅ 内存缓存支持 (LRU + TTL)
- ✅ 连接池支持 (Session 复用)
- ✅ DataFrame 批量插入
- ✅ ClickHouse 最新特性支持 (async_insert, Variant types, Projections)

## 安装

```bash
pip install ClickSQL
```

或从源码安装:

```bash
git clone https://github.com/sn0wfree/ClickSQL.git
cd ClickSQL
pip install -e .
```

### 依赖

- pandas >= 1.0.0
- numpy >= 1.20.0
- requests >= 2.20.0
- aiohttp >= 3.6.2
- orjson >= 3.6.0 (可选，性能提升 5-10 倍)
- fastmcp >= 2.0.0 (可选，用于 MCP 服务器，需要 Python 3.10+)

## 快速开始

### 基本连接

```python
from ClickSQL import ClickHouseTableNode

# 连接字符串格式: clickhouse://user:password@host:port/database
conn_str = "clickhouse://default:password@localhost:8123/my_database"
node = ClickHouseTableNode(conn_str)
```

### 执行查询

```python
# 简单查询
result = node.query("SELECT * FROM users LIMIT 10")

# 直接执行 SQL
result = node.execute("SELECT count(*) as cnt FROM orders")

# 使用缓存
result = node.execute("SELECT * FROM users", enable_memory_cache=True)
```

### 插入数据

```python
import pandas as pd

# 从 DataFrame 插入
df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'score': [95.5, 87.3, 92.1]
})

# 同步插入
node.insert_df(df, 'my_database', 'users', chunksize=100000)

# 并行插入
node.insert_df(df, 'my_database', 'users', parallel=True, max_workers=4)

# 异步插入
await node.insert_df_async(df, 'my_database', 'users')
```

## MCP 工具 (AI Agent 集成)

ClickSQL 支持 MCP 协议，可以作为工具为 AI Agent (如 Claude Desktop、VS Code Cline) 提供数据库服务。

### 安装 MCP 依赖

```bash
# Python 3.10+
pip install fastmcp

# Python 3.9 (MCP 服务器功能不可用，但工具函数仍可用)
pip install ClickSQL
```

### MCP 工具列表

| 工具 | 功能 |
|------|------|
| `execute_query` | 执行 SQL 查询 |
| `get_tables` | 获取数据库表列表 |
| `get_columns` | 获取表结构 |
| `get_databases` | 获取数据库列表 |
| `insert_dataframe` | 插入数据 |
| `create_table` | 创建表 |
| `get_table_count` | 获取表行数 |
| `table_exists` | 检查表是否存在 |
| `execute_async_query` | 异步执行查询 |

### 使用 MCP 工具函数

```python
from ClickSQL.mcp import tools

conn_str = "clickhouse://default:password@localhost:8123/my_db"

# 执行查询
result = tools.execute_query(conn_str, "SELECT * FROM users LIMIT 10")

# 获取表列表
tables = tools.get_tables(conn_str)

# 获取数据库列表
databases = tools.get_databases(conn_str)

# 获取表结构
columns = tools.get_columns(conn_str, 'my_db', 'users')

# 插入数据
data = [
    {'id': 1, 'name': 'Alice', 'score': 95.5},
    {'id': 2, 'name': 'Bob', 'score': 87.3}
]
result = tools.insert_dataframe(conn_str, data, 'my_db', 'users')

# 检查表是否存在
exists = tools.table_exists(conn_str, 'my_db', 'users')

# 获取表行数
count = tools.get_table_count(conn_str, 'my_db', 'users')
```

### 配置 Claude Desktop

1. 复制配置模板:

```bash
# Windows
copy examples\mcp_config.json %USERPROFILE%\.claude.json

# Linux/Mac
cp examples/mcp_config.json ~/.claude.json
```

2. 修改 `mcp_config.json`，配置您的 ClickHouse 连接:

```json
{
  "mcpServers": {
    "clicksql": {
      "command": "python",
      "args": ["-m", "ClickSQL.mcp.server"],
      "env": {
        "CLICKHOUSE_CONNECTION": "clickhouse://default:password@localhost:8123/my_database"
      }
    }
  }
}
```

3. 重启 Claude Desktop

4. 使用自然语言查询:
   - "查询 users 表的前 10 条数据"
   - "列出所有数据库"
   - "查看 orders 表的结构"
   - "插入一条新数据到 test 表"

### 运行 MCP 服务器

```bash
# Stdio 模式 (用于 Claude Desktop)
python -m ClickSQL.mcp.server

# SSE 模式 (HTTP，用于生产环境)
python -m ClickSQL.mcp.server sse
```

## 高级特性

### 1. 性能优化

#### orjson 加速

```python
# orjson 可提供 5-10 倍的 JSON 解析性能提升
# 安装: pip install orjson
```

#### 内存缓存

```python
# 启用内存缓存 (LRU + TTL)
result = node.execute("SELECT * FROM config", enable_memory_cache=True)

# 查看缓存状态
from ClickSQL import get_query_cache
cache = get_query_cache()
print(f"缓存条目数: {len(cache)}")

# 清理缓存
cache.clear()
```

#### 并行插入

```python
# 大数据量时启用并行插入
node.insert_df(
    large_dataframe, 
    'my_database', 
    'users', 
    parallel=True, 
    max_workers=8  # 默认 4
)
```

### 2. 异步支持

```python
import asyncio

async def async_query():
    # 客户端异步插入
    result = await node.insert_df_async(df, 'db', 'table')
    
    # 服务端异步插入 (async_insert)
    # 数据发送到服务器后立即返回，服务器自动批量写入
    result = node.insert_df_async_server(df, 'db', 'table', wait_for_async=False)
    
    # 异步执行
    result = node.query("SELECT * FROM table", async_mode=True)

# 运行异步代码
asyncio.run(async_query())
```

### 3. 连接池 (Session 复用)

```python
from ClickSQL import ClickHouseTableNode

# 默认启用 Session 复用，提升性能
node = ClickHouseTableNode("clickhouse://default:password@localhost:8123/my_db")

# 或者禁用 Session 复用
node = ClickHouseTableNode(
    "clickhouse://default:password@localhost:8123/my_db",
    reuse_session=False  # 每次请求创建新连接
)

# 也可以使用连接池版本
from ClickSQL.pool import PooledClickHouseTableNodeExt

pool_node = PooledClickHouseTableNodeExt(
    "clickhouse://default:password@localhost:8123/my_db",
    num_pools=10  # 连接池数量
)
```

### 4. ClickHouse Settings 参数

```python
# 传递 ClickHouse 设置参数
result = node.query(
    "SELECT * FROM large_table",
    settings={
        'max_threads': 8,
        'max_execution_time': 300,
        'optimize_throw_if_no_optimizes': 1
    }
)

# 使用 async_insert (服务端异步)
node.insert_df_async_server(
    df, 'db', 'table',
    wait_for_async=False  # 快速返回，由服务器处理写入
)
```

### 5. 表结构缓存

```python
# 首次调用会查询表结构，后续使用缓存
sql = node.get_sql('my_table', cols=['*'], use_cache=True)

# 清理特定表的缓存
node.clear_table_cache('my_table')

# 清理所有表缓存
node.clear_table_cache()
```

## API 参考

### ClickHouseTableNode

```python
# 创建节点
node = ClickHouseTableNode(conn_str)

# 查询
node.query(sql)           # 返回 DataFrame
node.execute(*sql)        # 执行单条或多条 SQL
node(sql)                # 简写形式

# 插入
node.insert_df(df, db, table, chunksize=100000, parallel=False)

# 元数据
node.tables              # 表列表
node.databases           # 数据库列表
node.columns             # 当前表列信息
```

### ClickHouseTableNodeExt

```python
node = ClickHouseTableNodeExt(conn_str)

# 高级查询
node.get_sql(db_table, cols=['*'], data_filter={...}, limit=100)

# 异步插入
await node.insert_df_async(df, db, table)

# 服务端异步插入
node.insert_df_async_server(df, db, table, wait_for_async=False)

# 表优化
node.optimize_table(db, 'table', final=True, deduplicate=True)

# Projections 管理
projections = node.list_projections(db, 'table')
node.create_projection(db, 'table', 'proj_name', 'SELECT col1, col2', order_by='col1')
node.drop_projection(db, 'table', 'proj_name')

# 系统日志查询
logs = node.system_logs('query_log', limit=100)
```

## 常见问题

### Q: 如何处理 DateTime 类型?

```python
# ClickHouse 返回的 DateTime 会自动转换为 pandas Datetime
result = node.query("SELECT created_at FROM users")
print(result['created_at'].dtype)  # datetime64[ns]
```

### Q: 如何处理 NULL 值?

```python
# ClickHouse 中的 NULL 会转换为 pandas NA
result = node.query("SELECT name FROM users WHERE name IS NULL")
# 使用 pandas 的 isna() 处理
result[result['name'].isna()]
```

### Q: 如何处理大数据查询?

```python
# 使用流式处理
for chunk in pd.read_sql("SELECT * FROM large_table", connection, chunksize=100000):
    process(chunk)

# 使用异步提高吞吐量
await node.insert_df_async(large_df, db, table, max_concurrent=10)
```

## 示例项目

See `examples/` directory:

- `mcp_example.py` - MCP 工具使用示例
- `mcp_config.json` - Claude Desktop 配置

## 测试

```bash
# 运行所有测试
python -m pytest test/ -v

# 运行 MCP 测试
python -m pytest test/test_mcp.py -v
```

## 许可证

MIT License

## 作者

sn0wfree
