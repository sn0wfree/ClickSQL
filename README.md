# ClickSQL: ClickHouse client for Humans 
 

 
Package information:

Records is a very simple, but powerful, library for making raw SQL queries to most relational databases.
 
ClickSQL is a smart client for ClickHouse database, which may help users to use ClickHouse more easier and smoother. 


more information for ClickHouse can be found at [here](http://clickhouse.tech)



## Installation

`pip install ClickSQL`

## Usage

```python
from ClickSQL import ClickHouseTableNode

conn_str = "clickhouse://default:test121231@99.99.9.9:8123/system"
ClickHouseTableNode(conn_str)

>>> connection test:  Ok.

``` 




