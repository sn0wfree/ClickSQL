# coding=utf-8
# coding=utf8
import responder

from ClickSQL.nodes.base import BaseSingleQueryBaseNode

conn_settings = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/system.columns'
# Tables =
api = responder.API()

# class GreetingResource:
#     def on_request(self, req, resp, *, greeting):   # or on_get...
#         resp.text = f"{greeting}, world!"
#         resp.headers.update({'X-Life': '42'})
#         resp.status_code = api.status_codes.HTTP_416
operator = BaseSingleQueryBaseNode(conn_settings, cols=[])

# @api.route("/{greeting}")
# class Service(object):
#     def __init__(self, conn_settings):
#         self.operator = BaseSingleFactorBaseNode(conn_settings, cols=[])


# def setup(self):
#     api.run(address=self.host, port=self.port, **self.api_kwargs)


# base_url = 'http://0.0.0.0:8279/'
# coding=utf8


# model_store_path = './static/model'


# @api.route("/")
# def hello_world(req, resp):
#     resp.text = "hello, world!"


FileExistsError_msg = 'data {} existed! please do not upload same data! '

#
# @api.route("/general/{sql}")
# def general(req, resp, *, sql):
#     pass


@api.route("/show_databases")
def show_datatbases(req, resp, ):
    res = operator('show databases').values.tolist()
    resp.status_code = 200
    resp.media = res


@api.route("/{db}/show_tables")
def db_table(req, resp, *, db):
    if db != 'query':
        res = operator(f'show tables from {db}').values.tolist()
        resp.status_code = 200
        resp.media = res
    else:
        resp.status_code = 400


@api.route("/{db}/{table}/show_columns")
def db_table(req, resp, *, db, table):
    res = operator(f'desc {db}.{table}').to_dict()
    resp.status_code = 200
    resp.media = res


@api.route("/query/{sql}")
def db_table(req, resp, *, sql):
    res = operator(sql).to_dict()
    resp.status_code = 200
    resp.media = res


if __name__ == '__main__':
    api.run(address='0.0.0.0', port=8237)
    pass
