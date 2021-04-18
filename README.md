# ClickSQL: ClickHouse client for Humans 
 

 
Package information:

 
ClickSQL is a smart client for ClickHouse database, which may help users to use ClickHouse more easier and pythonic. 
More information for ClickHouse can be found at [here](http://clickhouse.tech)



## Installation

`pip install ClickSQL`

## Usage
### Initial connection
to setup a database connection and send a heartbeat-check signal

```python
from ClickSQL import BaseSingleFactorTableNode

conn_str = "clickhouse://default:test121231@99.99.9.9:8123/system"
Node = BaseSingleFactorTableNode(conn_str)

>>> connection test:  Ok.

``` 

### Query
#### execute a SQL Query
```python
from ClickSQL import BaseSingleFactorTableNode

conn_str = "clickhouse://default:test121231@99.99.9.9:8123/system"
Node = BaseSingleFactorTableNode(conn_str)

Node('show tables from system limit 1')

>>> connection test:  Ok.
>>>                             name
>>> 0  aggregate_function_combinators
```

#### execute a Query without SQL
```python
from ClickSQL import BaseSingleFactorTableNode

factor = BaseSingleFactorTableNode(
        'clickhouse://default:default@127.0.0.1:8123/sample.sample',
        cols=['cust_no', 'product_id', 'money'],
        order_by_cols=['money asc'],
        money='money >= 100000'
    )


factor['money'].head(10)

>>> connection test:  Ok.
>>>        money
>>> 0  1000000.0
>>> 1  1000000.0
>>> 2  1000000.0
>>> 3  1000000.0
>>> 4  1000000.0
>>> 5  1000000.0
>>> 6  1000000.0
>>> 7  1000000.0
>>> 8  1000000.0
>>> 9  1000000.0


```


## Insert data
insert data into database by various ways
### Insert data via DataFrame
```python
from ClickSQL import BaseSingleFactorTableNode as factortable
import numpy as np
import pandas as pd
factor = factortable( 'clickhouse://default:default@127.0.0.1:8123/sample.sample'  )
db = 'sample'
table = 'sample'
df  = pd.DataFrame(np.random.random(size=(10000,3)),columns=['cust_no', 'product_id', 'money'])
factor.insert_df(df, db, table, chunksize=100000)
    

```

### Insert data via SQL(Inner)
```python
from ClickSQL import BaseSingleFactorTableNode as factortable

factor = factortable( 'clickhouse://default:default@127.0.0.1:8123/sample.sample'  )

factor("insert into sample.sample select * from other_db.other_table")
    

```

### Create table

#### Create table by SQL
```python
from ClickSQL import BaseSingleFactorTableNode

conn_str = "clickhouse://default:test121231@99.99.9.9:8123/system"
Node = BaseSingleFactorTableNode(conn_str)

Node('create table test.test2 (v1 String, v2 Int64, v3 Float64,v4 DataTime) Engine=MergeTree() order by v4')
```

#### Create table by DataFrame
```python
from ClickSQL import BaseSingleFactorTableNode
import numpy as np
import pandas as pd

conn_str = "clickhouse://default:test121231@99.99.9.9:8123/system"
Node = BaseSingleFactorTableNode(conn_str)
db = 'test'
table = 'test2'


df_or_sql_or_dict  = pd.DataFrame(np.random.random(size=(10000,2)),columns=['v1', 'v3'])
df_or_sql_or_dict['v2'] =1
df_or_sql_or_dict['v4'] =pd.to_datetime('2020-01-01 00:00:00')

Node.create( db,  table,  df_or_sql_or_dict,    key_cols=['v4'],)
```


### Contribution
Welcome to improve this package or submit an issue or any others

## Author
sn0wfree




# Plan
## Available functions or properties
1. get data from clickhouse
2. insert data into clickhouse
3. create 
4. alter
5. execute standard SQL and transform into DataFrame(Auto)
3. able to execute select query 
4. able to execute insert query 
5. no require clickhouse-client
6. auto create table sql
7. can execute explain query
8. Insert Data via DataFrame
3. alter function & drop function

## In Process
2. create a pandas_liked executable function, which can compatible with pandas 
9. distributed query（query+insert）


## schedule
1. ORM
4. can execute user role query
5. create analysis component
6. auto report system
7. table register system
8. data manager system
8. meta data manager



