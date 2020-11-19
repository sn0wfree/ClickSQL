# coding=utf-8
# coding=utf-8
from ClickSQL.utils.file_cache import file_cache
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from collections import namedtuple
from functools import partial
from ClickSQL.utils.boost_up import boost_up

# fig = plt.figure()
# sns.set_palette("GnBu_d")
# sns.set_style('whitegrid')


evet_settings = namedtuple('event', ('before_event_window', 'event_window', 'prefix_event', 'gap'))

# fig = plt.figure()
# sns.set_palette("GnBu_d")
# sns.set_style('whitegrid')

import yfinance as yf

##没有安装yahoo finance的话可以这样安装：
## !pip install yfinance

stock_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
              'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC']
## AMZN: 亚马逊,  AAPL: 苹果, TSLA: 特斯拉, GE: 通用电气, GILD: 吉利德（做瑞德西韦那个公司）,
## BA: 波音, NFLX: 网飞, MS: 大摩,
## LNVGY: 联想 BABA: 阿里巴巴, LK: 瑞幸, JOBS: 前程无忧, CEO: 中海油, TSM: 台积电, JD: 京东,
## ^GSPC: SP500
# prices = pd.DataFrame()
# ## get data
# for stock in stock_list:
#     symbol = stock
#     tickerData = yf.Ticker(symbol)
#     price_data = tickerData.history(period='1d', start='2019-1-1', end='2020-4-1')['Close']
#     prices[stock] = price_data
# return_df = prices.pct_change().dropna()
firm_list = stock_list[:-1]
date = ['2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13',
        '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23']
date_df = pd.DataFrame(date, index=firm_list, columns=['Date'])
date_df.index.name = 'CompanyName'


#

# return_df['RF'] = 0.0065
# return_df['Mkt_RF'] = return_df['^GSPC'] - return_df['RF']


class EventStudy(object):
    @classmethod
    @file_cache()
    def cal_ar(cls, return_df: pd.DataFrame, event_dict: dict, date='Date', rf='RF',
               factors=['Mkt_RF'], formula="{stock} ~ {rf} + {factors_str}",
               event_info: tuple = (250, 20, 10, 1), ar_only=True, boost=True):
        func = partial(cls.cal_ar_single,
                       date=date,
                       rf=rf, factors=factors,
                       formula=formula,
                       event_info=event_info,
                       ar_only=ar_only)

        tasks = ((return_df, event_happen_day, stock) for stock, event_happen_day in event_dict.items())

        if boost:
            holder = boost_up(func, tasks, star=True)
        else:
            holder = [func(return_df, event_happen_day, stock_return=stock) for _, event_happen_day, stock in tasks]
        return pd.concat(holder, axis=1)

    @staticmethod
    def cal_ar_single(return_df: pd.DataFrame, event_happen_day: str, stock: str,
                      date='Date', rf='RF', factors=['Mkt_RF'], formula="{stock} ~ {rf} + {factors_str}",
                      event_info: tuple = (250, 20, 10, 1), ar_only=True):
        event_set = evet_settings(*event_info)

        variables = [date, stock, rf] + factors
        suffix_event = event_set.event_window - event_set.prefix_event

        event_index = int(return_df[return_df[date] == str(event_happen_day)].index.values)
        estimation_df = return_df.loc[event_index - (
                event_set.before_event_window + event_set.prefix_event + event_set.gap): event_index - (
                event_set.prefix_event + event_set.gap), variables]

        f1 = formula.format(stock=stock, rf=rf, factors_str='+'.join(factors))
        print(f1)
        models = sm.OLS.from_formula(f1, data=estimation_df).fit()
        params = models.params
        beta_Mkt = params[factors]
        alpha = params["Intercept"]
        bse = models.bse

        # expected returns for each firm in the estimation window
        event_df = return_df.loc[event_index - event_set.prefix_event: event_index + suffix_event,
                   variables].reset_index(drop=True)

        event_df[f'{stock}_er'] = np.dot(event_df[factors].values, beta_Mkt) + alpha
        event_df[f'{stock}_ar'] = event_df.eval(f'{stock} - {stock}_er')
        event_df.index = event_df.index - event_set.prefix_event
        if ar_only:
            output_cols = [f'{stock}_ar']
            #             resistd = event_df[f'{stock}_abnormal_return'].std() / corp_count

            #             resistd = event_df[f'{stock}_abnormal_return'].std() / corp_count

            return event_df[output_cols]
        else:
            return event_df

    @staticmethod
    def cal_aar(ar_df: pd.DataFrame):
        aar_df = pd.DataFrame(ar_df.mean(axis=1), columns=['aar'])
        return aar_df

    @staticmethod
    def cal_car(ar_df: pd.DataFrame):
        return ar_df.sum(axis=1)
    @staticmethod
    def cal_caar(aar_df: pd.DataFrame):
        return aar_df.sum()


#
# def eventstudy(returndata, eventdata, stocklist):
#     """
#     returndata: is a dataframe with the market returns of the different firms
#     eventdata: eventdata for the different firms
#     stocklist: a list of the firms involved in the analysis
#
#     Returns:
#     abnreturn: a dictionary of the abnormal returns for each firm in their respective eventwindows -/+20
#     """
#     abnreturn = {}  # abnormal returns on the event window
#     returndata = returndata.reset_index()
#     Bse = []
#
#     event_window = 20
#     prefix_event = 10
#     suffix_event = event_window - prefix_event
#     before_event_window = 250
#     formula = '{stock} ~ {rf} + {factors}'
#
#     for stock in stocklist:
#         eventindex = int(returndata[returndata['Date'] == str(eventdata.at[stock, 'Date'])].index.values)
#         print(eventindex)
#         event_df = returndata.loc[eventindex - prefix_event: eventindex + suffix_event, ['Date', stock, 'RF', 'Mkt_RF']]
#         estimation_df = returndata.loc[
#                         eventindex - (before_event_window + prefix_event): eventindex - (prefix_event + 1),
#                         ["Date", stock, 'RF', 'Mkt_RF']]
#         formula = formula.format()
#         beta_Mkt = sm.OLS.from_formula(formula, data=estimation_df).fit().params["Mkt_RF"]
#
#         alpha = sm.OLS.from_formula(formula, data=estimation_df).fit().params["Intercept"]
#
#         standard_error = sm.OLS.from_formula(formula, data=estimation_df).fit().bse
#
#         Bse.append(standard_error)
#         print("{}, beta_Mkt= {},alpha= {}".format(stock, beta_Mkt, alpha))
#
#         # expected returns for each firm in the estimation window
#         expectedreturn_eventwindow = ((event_df[['Mkt_RF']].values * beta_Mkt) + alpha)
#
#         # abnormal returns on the event window - AR
#
#         abnormal_return = event_df[stock].values - list(expectedreturn_eventwindow.flatten())
#         abnreturn[stock] = abnormal_return
#
#     abnormalreturns_df = pd.DataFrame(abnreturn)
#     abnormalreturns_df.index = abnormalreturns_df.index - 10
#     return abnormalreturns_df
#
#
# def CAR_se(Abnormal_return, stock_list):
#     """
#     To get the standard error of Cumulative Abnormal Return for each stock
#     Input: the Abnormal Return datafram or matrix, a list of company names
#     Output: a dataframe of cumulative standard error for each stock
#     """
#     residual_sigma_single = pd.DataFrame()
#     residual_sigma_cum_single = pd.DataFrame()
#     resi_single = []
#     d = {}
#     for x in stock_list:
#         resistd = abnormalreturns_df[x].std() / 15
#         d.update({x: resistd})
#
#     residual_sigma_single = pd.DataFrame(d, index=Abnormal_return.index)
#     residual_sigma_cum_single = np.sqrt(residual_sigma_single.cumsum())
#     se_cum_single = np.sqrt(((residual_sigma_cum_single ** 2) / 15))
#
#     return se_cum_single
#
#
# def CAAR_se(Abnormal_return, stock_list):
#     """
#     To get the standard error of Cumulative Average Abnormal Return
#     Input: the Abnormal Return datafram or matrix, a list of company names
#     Output: a list of cumulative standard error
#     """
#     residual_sigma = pd.DataFrame()
#     resi = []
#     d = {}
#     for x in stock_list:
#         resistd = abnormalreturns_df[x].std() / 15
#         d.update({x: resistd})
#
#     residual_sigma = pd.DataFrame(d, index=Abnormal_return.index)
#     residual_sigma_cum = np.sqrt(residual_sigma.cumsum())
#     se_cum = np.sqrt(((residual_sigma_cum ** 2) / 15).mean(axis=1))
#
#     return se_cum
#
#
# abnormalreturns_df = eventstudy(returndata=return_df, eventdata=date_df,
#                                 stocklist=firm_list)
# # plt.figure(figsize=(24, 12))
# # for i in range(1, 16):
# #     plt.subplot(4, 4, i)
# #     abnormalreturns_df[abnormalreturns_df.columns[i - 1]].plot()
# #     plt.xlabel('Event Window')
# #     plt.ylabel('Return')
# #     plt.axhline(y=(np.sqrt((abnormalreturns_df.iloc[:, i - 1].std() ** 2 / 21)) * 1.96), color='red', linestyle='--')
# #     plt.axhline(y=(np.sqrt((abnormalreturns_df.iloc[:, i - 1].std() ** 2 / 21)) * -1.96), color='red', linestyle='--')
# #     plt.title(abnormalreturns_df.columns[i - 1])
#
# mean_AAR = abnormalreturns_df.mean(axis=1)
# var_AAR = (abnormalreturns_df.std()) ** 2
# var_matrix = pd.DataFrame(var_AAR)
# var_matrix = var_matrix.T
# var_AAR = sum(var_matrix.iloc[0]) / 15 ** 2
# Std_AAR = np.sqrt(var_AAR)
# mean_AAR.plot()
# # plt.axhline(y=Std_AAR * 1.96, color='red', linestyle='--')
# # plt.axhline(y=Std_AAR * -1.96, color='red', linestyle='--')
#
# # CAR和CAAR
#
# se_cum_single = CAR_se(abnormalreturns_df, firm_list)
# CAR_df = abnormalreturns_df.cumsum()
# # plt.figure(figsize=(24, 12))
# # for i in range(1, 16):
# #     plt.subplot(4, 4, i)
# #     CAR_df[CAR_df.columns[i - 1]].plot()
# #     plt.plot(se_cum_single.iloc[:, i - 1] * 1.96, color='red', linestyle='--')
# #     plt.plot(se_cum_single.iloc[:, i - 1] * -1.96, color='red', linestyle='--')
# #     plt.xlabel('Event Window')
# #     plt.ylabel('CAR')
# #     plt.title(CAR_df.columns[i - 1])
#
# se = CAAR_se(abnormalreturns_df, firm_list)
# Var_AAR = ((CAR_df.mean(axis=1)) ** 2) / 15
# Std_AAR = np.sqrt(Var_AAR)
# # CAAR
# CAAR = mean_AAR.cumsum()
# # Plot CAAR
# CAAR.plot(figsize=(12, 8))
# # plt.xlabel("Event Window")
# # plt.plot(se * 1.96, color='red', linestyle='--')
# # plt.plot(se * -1.96, color='red', linestyle='--')
# # plt.ylabel("Cumulative Return")
# # plt.title("Cumulative Average Abnormal Return")

if __name__ == '__main__':
    return_df = pd.read_pickle('data.pkl').reset_index()
    event_dict = date_df.to_dict()['Date']
    ar_df = EventStudy.cal_ar(return_df,
                              event_dict=event_dict,
                              date='Date',
                              rf='RF', factors=['Mkt_RF'],
                              formula="{stock} ~ {rf} + {factors_str}",
                              event_info=(250, 20, 10, 1), ar_only=True, boost=True)
    aar_df = EventStudy.cal_aar(ar_df)
    car_df = EventStudy.cal_car(ar_df)
    print(car_df)
    pass
