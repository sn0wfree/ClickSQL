# coding=utf-8

from ClickSQL.eventstudy.es_base import DefaultModel
import statsmodels.api as sm
import numpy as np
import pandas as pd


class MarketModel(DefaultModel):
    @staticmethod
    def cal_ar(estimation_df, event_df, stock: str, formula: str = "{stock} ~ 1 + {factors}", factors=['Mkt_RF']):
        """

        :param estimation_df:  pd.DataFrame
        :param event_df:  pd.DataFrame
        :param stock:
        :param formula:
        :param factors:
        :return:
        """
        f1 = formula.format(stock=stock, factors="+".join(factors))
        # print(f1)
        models = sm.OLS.from_formula(f1, data=estimation_df).fit()
        params = models.params
        beta_Mkt = params[factors]
        alpha = params["Intercept"]
        bse = models.bse
        event_df[f'{stock}_er'] = np.dot(event_df[factors].values, beta_Mkt) + alpha
        # event_df[f'{stock}_ar'] = event_df.eval(f'{stock} - {stock}_er')
        return event_df.eval(f'{stock} - {stock}_er')


if __name__ == '__main__':
    from ClickSQL.eventstudy.es_base import EventStudy

    stock_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
                  'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC']
    firm_list = stock_list[:-1]
    # date = ['2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13',
    #         '2020-03-13', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23',
    #         '2020-01-23']
    # date_df = pd.DataFrame(date, index=firm_list, columns=['Date'])
    # date_df.index.name = 'CompanyName'
    #
    return_df = pd.read_csv('data.csv')
    # event_dict = date_df.to_dict()['Date']
    event_dict = {'AMZN': ['2020-03-13', '2020-01-23'], 'AAPL': '2020-03-13', 'TSLA': '2020-03-13', 'GE': '2020-03-13',
                  'GILD': '2020-03-13', 'BA': '2020-03-13', 'NFLX': '2020-03-13', 'MS': '2020-03-13',
                  'LNVGY': '2020-01-23', 'BABA': '2020-01-23', 'LK': '2020-01-23', 'JOBS': '2020-01-23',
                  'CEO': '2020-01-23', 'TSM': '2020-01-23', 'JD': '2020-01-23'}
    # event_dict['t'] = 1
    es = EventStudy(model=MarketModel, event_info=(250, 10, 5, 1))
    res = es(return_df, event_dict=event_dict, date='Date', factors=['Mkt_RF'], )
    # np.sqrt(ar.iloc[:,9-1].std()**2/17)*1.96
    # print(es.aar)

    print(res)
