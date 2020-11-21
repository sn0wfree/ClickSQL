# coding=utf-8
import pandas as pd
from collections import namedtuple

event_tuple = namedtuple('event', ('before_event_window', 'event_window', 'prefix_event', 'gap'))


# firm_list = stock_list[:-1]
# date = ['2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13',
#         '2020-03-13', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23',
#         '2020-01-23']
# date_df = pd.DataFrame(date, index=firm_list, columns=['Date'])
# date_df.index.name = 'CompanyName'
# #
# return_df = pd.read_csv('data.csv')
# event_dict = date_df.to_dict()['Date']

class Data(object):
    @staticmethod
    def get_stock_list():
        stock_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
                      'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC']
        firm_list = stock_list[:-1]
        return firm_list

    @staticmethod
    def get_event_list_single(stock_list=['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
                                          'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC'],
                              event_type='up'):
        stock_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
                      'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC']
        firm_list = stock_list[:-1]
        date = ['2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13',
                '2020-03-13', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23',
                '2020-01-23']

        date_df = pd.DataFrame(date, index=firm_list, columns=['Date'])
        date_df.index.name = 'CompanyName'
        event_dict = date_df.to_dict()['Date']
        return event_dict

    @staticmethod
    def get_return_data(stock_list=['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
                                    'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC'], start='2019-05-20',
                        end='2020-03-31'):
        # 2020-03-31
        stock_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
                      'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC']
        firm_list = stock_list[:-1]
        return_df = pd.read_csv('data.csv')
        return return_df



# cal cumsum change

if __name__ == '__main__':
    eval_window = 10
    event_info: tuple = (250, 20, eval_window, 1)

    pass
