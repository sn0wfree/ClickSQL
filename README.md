# ClickSQL: ClickHouse client for Humans 
 

 
Package information:

Records is a very simple, but powerful, library for making raw SQL queries to most relational databases.
 
ClickSQL is a smart client for ClickHouse database, which may help users to use ClickHouse more easier and smoother. 


more information for ClickHouse can be found at [here](http://clickhouse.tech)



## Installation

`pip install ClickSQL`

## Usage
### initial connection

#### *Approach 1*
```python
from ClickSQL import ClickHouseTableNode

conn_str = "clickhouse://default:test121231@99.99.9.9:8123/system"
ct = ClickHouseTableNode(conn_str)

>>> connection test:  Ok.

``` 

#### *Approach 2*
```python
from ClickSQL import ClickHouseTableNode



conn_str = {'name':'clickhouse','host':'99.99.9.9','port':8123,'user':'default',
            'password':'test121231','database':'system'}
ct = ClickHouseTableNode(**conn_str)

>>> connection test:  Ok.

``` 
### Query

```python
from ClickSQL import ClickHouseTableNode

conn_str = "clickhouse://default:test121231@99.99.9.9:8123/system"
ct = ClickHouseTableNode(conn_str)

ct.query('show tables from system limit 1')

>>> connection test:  Ok.
>>>                             name
>>> 0  aggregate_function_combinators
```

### Contribution
there is welcome to do more work to improve this package more convenient

## Author
sn0wfree

## functions
1. get data from clickhouse
2. insert data into clickhouse
3. create 
4. alter



