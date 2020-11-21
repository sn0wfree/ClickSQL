# coding=utf-8
# coding=utf8
import requests
from ClickSQL.nodes.ch_factor import BaseSingleFactorBaseNode

conn_settings = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/system.columns'
Tables = BaseSingleFactorBaseNode(conn_settings, cols=[], )

base_url = 'http://0.0.0.0:8279/'
# coding=utf8
import time, os
import responder
from ClickSQL.utils.uuid_generator import uuid_hash
# from tools.save_link import insert_data as insert_data_sync
# from auto_ml.core.parameter_parser import ModelStore
# from auto_ml.core.aml import AML

data_store_path = './static/data/'

link_file = './static/link_file.sqlite'
tableName = 'link'
model_store_path = './static/model/'

api = responder.API()

# model_store_path = './static/model'


# @api.route("/")
# def hello_world(req, resp):
#     resp.text = "hello, world!"


FileExistsError_msg = 'data {} existed! please do not upload same data! '


class FileExistsError(Exception):
    def __init__(self, msg):
        self.msg = msg


@api.route("/upload_file")
async def upload_file(req, resp):
    @api.background.task
    def store_data(content, path, filename):
        """
        store data for model
        :param content: binary data
        :param path:  store path
        :param filename:  filename
        :return:
        """
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(content)
        else:
            raise FileExistsError(FileExistsError_msg.format(filename))

    data = await req.media(format='files')
    # data = req.media(format='files')

    data_uuid = uuid_hash(str(data))

    filename = data_uuid  # data['file']['filename']
    content = data['file']['content']
    path = data_store_path + '{}'.format(filename)

    msg = FileExistsError_msg.format(filename)
    if not os.path.exists(path):
        print(filename)
        store_data(content, path, filename)
        resp.media = {'dataid': filename, 'status': 'good', 'store_status': 'ok'}
        resp.status_code = api.status_codes.HTTP_200
    else:
        resp.media = {'dataid': 'error', 'status': 'duplicates', 'store_status': msg}
        resp.status_code = api.status_codes.HTTP_409


@api.route("/check_file/{dataid}")
async def check_file(req, resp, *, dataid):
    if os.path.exists(data_store_path + dataid):
        resp.text = f'{dataid} found!'
        resp.status_code = api.status_codes.HTTP_200

    else:
        resp.text = f"{dataid} not found!"
        resp.status_code = api.status_codes.HTTP_404


@api.route("/AutoML/{dataid}")
class AutoML(object):
    # def on_get(self, req, resp, *, dataid):
    #     if dataid == 'get_supported_model':
    #         parameters = self.parse_parameters(req.params)

    def on_request(self, req, resp, *, dataid):  # or on_get...
        @api.background.task
        def insert_data( sqlfile, model_id, data_id, model_path=model_store_path, data_path=data_store_path,
                              tableName=tableName):
            insert_data_sync(sqlfile, model_id, data_id, model_path=model_path, data_path=data_path,
                             tableName=tableName)

        parameters = self.parse_parameters(req.params)
        if dataid == 'getParams':
            if 'type' in parameters:
                result = AML.get_supported_model(parameters['type'], raiseError=False)

                resp.text = f"{result}"
                resp.status_code = api.status_codes.HTTP_200
            else:
                resp.text = f"{dataid}, wrong parameters! Please retry! "
                resp.status_code = api.status_codes.HTTP_416

        else:
            print(parameters, type(parameters))
            if os.path.exists(data_store_path + dataid):
                dataset_dict = self._load_dataset(data_store_path + dataid)
                result = self.run_program(parameters, dataset_dict)

                model_uuid = uuid_hash(str(ModelStore._save_in_memory(result)))  # generator model id
                if not os.path.exists(data_store_path + model_uuid):
                    ModelStore._save(result, model_store_path + model_uuid)

                insert_data(link_file,
                            model_uuid,
                            dataid,
                            model_path=model_store_path,
                            data_path=data_store_path,
                            tableName=tableName)

                resp.text = f"{result}"
                # resp.headers.update({'X-Life': '42'})
                resp.status_code = api.status_codes.HTTP_200
            else:
                resp.text = f"{dataid}, No dataset found! Please upload data first!"
                resp.status_code = api.status_codes.HTTP_416

    @staticmethod
    def _load_dataset(dataset):

        strings = ModelStore._force_read(dataset)
        return ModelStore._force_read_from_string(strings)

        pass

    @classmethod
    def run_program(cls, parameters, dataset):

        m = AML.run(parameters, dataset)
        # dataset_dict = cls._load_dataset(dataset)

        return m

    @staticmethod
    def parse_parameters(params):
        pa = {}
        for key, values in params.items():
            if values == 'Null':
                values = None
            elif values == '[]':
                values = []
            elif values.isnumeric():
                values = int(values)
            else:
                pass
            pa[key] = values

        return pa


# @api.route("/auto_ml")
# def auto_ml(req, resp):
#     paras = req.media()
#
#     print(paras, paras['parameters'])
#     resp.media = {'filename': 's', 'status': 'good', 'store_status': 'ok'}
#     pass


# @api.route("/incoming")
# async def receive_incoming(req, resp):
#     @api.background.task
#     def process_data(data):
#         """Just sleeps for three seconds, as a demo."""
#         time.sleep(3)
#
#     # Parse the incoming data as form-encoded.
#     # Note: 'json' and 'yaml' formats are also automatically supported.
#     data = await req.media()
#
#     # Process the data (in the background).
#     process_data(data)
#
#     # Immediately respond that upload was successful.
#     resp.media = {'success': True}
#

if __name__ == '__main__':
    api.run(address='0.0.0.0', port=8279)
    pass
# if __name__ == '__main__':
#     columns = Tables.fetch_all()
#     print(columns)
#     pass
