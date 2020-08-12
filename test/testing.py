# coding=utf-8
import pandas as pd
import numpy as np
from ClickSQL.core.clickhouse import ClickHouseTableNode
if __name__ == '__main__':
    node = ClickHouseTableNode(
        **{'host': '47.104.186.157', 'port': 8123, 'user': 'default', 'password': 'Imsn0wfree',
           'database': 'EDGAR_LOG'})
    df1 = node.tables
    print(df1)
    pass


