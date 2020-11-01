# 规划
## 目前已经实现了的功能
1. 访问Clickhouse服务
2. 执行标准的SQL语句,并且直接转为dataframe
3. 能够执行select 查询语句
4. 能够执行insert 插入语句
5. 独立python组件，不需要安装clickhouse-client


## 未来需要构造的功能
1. 自动建表语句
2. 构造类pandas的执行语言，可以无缝切换pandas部分接口
3. 能够执行explain语句
4. 能够执行用户权限控制
5. 自动构造分析型组件
6. 自动报表系统
7. 能够自动化更新系统(表管理系统)
8. 元数据管理
9. 分布式查询（query+insert）

