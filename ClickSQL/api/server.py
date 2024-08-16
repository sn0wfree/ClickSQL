# coding=utf-8

import io
from typing import Union

import pandas as pd
from fastapi import FastAPI  # 1. 导入 FastAPI
from fastapi.responses import Response

from ClickSQL.nodes.base import BaseSingleFactorTableNode
from ClickSQL.utils.lru_cache import lru_cache

src = "clickhouse://default:Imsn0wfree@0.0.0.0:8123/system"
conn = BaseSingleFactorTableNode(src)


def parse_df(content):
    # 将文件内容转换为 BytesIO 对象
    buffer = io.BytesIO(content)
    return pd.read_parquet(buffer)


# @lru_cache(timeout=60)
def send_df(df):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    file_obj = buffer.getvalue()
    return file_obj


@lru_cache(timeout=60)
def query(sql):
    return conn(sql)


app = FastAPI()  # 2. 创建一个 FastAPI 实例


@app.get('/')  # 3. 创建一个路径操作
async def hello():  # 4. 定义路径操作函数
    return {'message': 'Hello World!'}  # 5. 返回内容


@app.get("/sql/{sql}")
def sql_appoarch(sql: str, q: Union[str, None] = None):
    nav = query(sql)
    if nav.empty:
        return None
    else:
        parquet_data = send_df(nav)
        return Response(content=parquet_data)


if __name__ == '__main__':
    pass
