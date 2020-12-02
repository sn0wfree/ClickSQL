# coding=utf-8
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.api as sm
from scipy.stats import t

from ClickSQL.utils import boost_up, cached_property
import warnings

# fig = plt.figure()
# sns.set_palette("GnBu_d")
# sns.set_style('whitegrid')


event_tuple = namedtuple('event', ('before_event_window', 'event_window', 'post_event_window', 'gap'))


# fig = plt.figure()
# sns.set_palette("GnBu_d")
# sns.set_style('whitegrid')

# import yfinance as yf

## 没有安装yahoo finance的话可以这样安装：
## !pip install yfinance


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


#

# return_df['RF'] = 0.0065
# return_df['Mkt_RF'] = return_df['^GSPC'] - return_df['RF']

class EventDataKeyError(KeyError): pass


class DefaultModel(object):
    @staticmethod
    def check_type(name: str, data: object, data_type: object):
        if isinstance(data, data_type):
            pass
        else:
            raise TypeError(f"{name} got wrong type! only accept {data_type}")

    @classmethod
    def real_run(cls, *args, **kwargs) -> pd.Series:
        estimation_df, event_df, stock = args
        cls.check_type('estimation_df', estimation_df, pd.DataFrame)
        cls.check_type('event_df', event_df, pd.DataFrame)
        cls.check_type('stock', stock, str)
        if 'formula' in kwargs.keys():
            cls.check_type('formula', kwargs['formula'], str)
        cls.check_type('factors', kwargs['factors'], list)
        return cls.cal_ar(*args, **kwargs)

    @staticmethod
    def cal_ar(*args, **kwargs):
        raise ValueError('cal_ar have not been defined!')


class EventStudyUtils(object):
    @staticmethod
    def split_event(return_df: pd.DataFrame, event_happen_day: str, stock, date='Date', factors=['Mkt_RF'],
                    event_info: tuple = (250, 20, 10, 1), detect=True):
        variables = [date, stock] + factors
        if detect:
            if event_info[-2] < 0:
                if event_info[-1] > abs(event_info[-2]) + 1:
                    pass
                else:
                    warnings.warn('estimation period covered event date! will shift estimation period!')
                    event_info = (event_info[0], event_info[1], event_info[2], abs(event_info[-2]) + 1)

        event_set = event_tuple(*event_info)
        post_event_window = event_set.post_event_window
        after_event_window = event_set.event_window - post_event_window
        try:
            data = return_df[variables]
        except KeyError as e:
            raise EventDataKeyError(e)
        event_index_loc = int((pd.to_datetime(data[date]) - pd.to_datetime(event_happen_day)).abs().sort_values().index[0])
        estimation_df = data.loc[event_index_loc - (
                event_set.before_event_window + event_set.post_event_window + event_set.gap): event_index_loc - (
                event_set.post_event_window + event_set.gap), variables]
        event_df = data.loc[event_index_loc - event_set.post_event_window: event_index_loc + after_event_window,
                   variables].reset_index(drop=True)

        event_df.index = event_df.index - event_set.post_event_window

        return estimation_df, event_df

    @classmethod
    def cal_ar_single(cls, return_df: pd.DataFrame, event_happen_day: str, stock: str,
                      date='Date', factors=['Mkt_RF'], event_info: tuple = (250, 20, 10, 1), ar_only=True,
                      model=DefaultModel):
        # event_set = event_tuple(*event_info)
        #
        # variables = [date, stock, rf] + factors
        # data = return_df[variables]
        # data[stock] = data.eval(f'{stock} - {rf}')
        # after_event_window = event_set.event_window - event_set.post_event_window
        #
        # event_index = int(data[data[date] == str(event_happen_day)].index.values)
        # estimation_df = data.loc[event_index - (
        #         event_set.before_event_window + event_set.post_event_window + event_set.gap): event_index - (
        #         event_set.post_event_window + event_set.gap), variables]

        estimation_df, event_df = cls.split_event(return_df, event_happen_day, stock, date=date,
                                                  event_info=event_info, factors=factors)

        # expected returns for each firm in the estimation window
        # event_df = data.loc[event_index - event_set.post_event_window: event_index + after_event_window,
        #            variables].reset_index(drop=True)
        if issubclass(model, DefaultModel):
            ar_series = model.real_run(estimation_df, event_df, stock, factors=factors)
        else:
            raise TypeError(f'model got wrong definition: {model.__name__}! should be a DefaultModel-liked class')

        # event_df[f'{stock}_er'] = np.dot(event_df[factors].values, beta_Mkt) + alpha
        # event_df[f'{stock}_ar'] = event_df.eval(f'{stock} - {stock}_er')
        # event_df.index = event_df.index - event_set.post_event_window
        if ar_only:
            # output_cols = [f'{stock}_ar']
            return pd.DataFrame(ar_series)
        else:
            event_df[f'{stock}_ar'] = ar_series
            return event_df

    @classmethod
    def cal_residual(cls, return_df: pd.DataFrame, event_dict: dict, date='Date',
                     factors=['Mkt_RF'], model=DefaultModel,
                     event_info: tuple = (250, 20, 10, 1), ar_only=True, boost=True):

        """
        计算异常收益率并加总异常收益率(CAR)
[公式] 计算的是股票 [公式] 在第 [公式] 天的异常收益率，为了研究事件对整体证券定价的影响，还需要计算平均异常收益率 [公式] 和累积异常收益率 [公式] 。
通常而言，平均异常收益率是针对某一时点、对所有公司的异常收益率求平均，计算方式如下所示：

[公式]

        :param return_df:
        :param event_dict:
        :param date:
        :param factors:
        :param model:
        :param event_info:
        :param ar_only:
        :param boost:
        :return:
        """
        func = partial(cls.cal_ar_single, date=date, factors=factors, model=model,
                       event_info=event_info, ar_only=ar_only)

        tasks = ((return_df, event_happen_day, stock) for stock, event_happen_day in event_dict.items())

        if boost:
            holder = boost_up(func, tasks, star=True)
        else:
            holder = [func(return_df, event_happen_day, stock=stock) for _, event_happen_day, stock in tasks]
        return pd.concat(holder, axis=1)

    # @staticmethod
    # def cal_aar(ar_df: pd.DataFrame) -> pd.Series:
    #     """
    #
    #     :param ar_df:
    #     :return:
    #     """
    #     aar_series = pd.DataFrame(ar_df.mean(axis=1), columns=['aar'])
    #
    #     return aar_series

    # @staticmethod
    # def cal_car(ar_df: pd.DataFrame) -> pd.Series:
    #     """
    #
    #     :param ar_df:
    #     :return:
    #     """
    #
    #     # car_series = pd.DataFrame(ar_df.sum(axis=0), columns=['car'])
    #     return ar_df.cumsum()

    # @staticmethod
    # def cal_caar(aar_df: pd.DataFrame):
    #     """
    #
    #     :param aar_df:
    #     :return:
    #     """
    #
    #     return aar_df.sum()


class EventStudy(EventStudyUtils):
    def __call__(self, return_df: pd.DataFrame, event_dict: dict, date='Date', factors=['Mkt_RF'],
                 event_info: tuple = (250, 20, 10, 1), ):
        data, new_event_dict = self.detect_multi_event_point(return_df, event_dict)
        self.data = data
        self.event_dict = new_event_dict
        self.event_info = event_info
        self.cols = dict(date=date, factors=factors)
        return self.result

    def __init__(self, return_df: (None, pd.DataFrame) = None, event_dict: (dict, None) = None, date='Date',
                 factors=['Mkt_RF'], model=DefaultModel, event_info: tuple = (250, 20, 10, 1), ar_only=True,
                 boost=False):
        """
        >>>  es = EventStudy(return_df,event_dict=event_dict,date='Date', factors=['Mkt_RF'],
                             formula="{stock} ~ 1 + {factors_str}",event_info=(250, 10, 5, 1))

        >>> es.result
                      ar    var_ar       car   var_car t_statistic  p_values
            -5 -0.005464  0.000102 -0.005464  0.000102 -0.540883  0.294536
            -4 -0.002986  0.000102 -0.008450  0.000204 -0.591467  0.277372
            -3 -0.004987  0.000102 -0.013437  0.000306 -0.767973  0.221615
            -2 -0.011591  0.000102 -0.025028  0.000408 -1.238760  0.108300
            -1 -0.006547  0.000102 -0.031576  0.000510 -1.397834  0.081703
            0  -0.020056  0.000102 -0.051632  0.000612 -2.086547  0.018973
            1  -0.005131  0.000102 -0.056763  0.000714 -2.123751  0.017339
            2  -0.027366  0.000102 -0.084129  0.000816 -2.944349  0.001771
            3   0.004946  0.000102 -0.079183  0.000918 -2.612757  0.004764
            4   0.007692  0.000102 -0.071491  0.001021 -2.237908  0.013056
            5  -0.001437  0.000102 -0.072928  0.001123 -2.176649  0.015223




        :param return_df:
        :param event_dict:
        :param date:

        :param factors:

        :param event_info:
        :param ar_only:
        :param boost:
        """

        data, new_event_dict = self.detect_multi_event_point(return_df, event_dict)
        self.data = data
        self.event_dict = new_event_dict
        self.cols = dict(date=date, factors=factors)
        self.model = model
        self.event_info = event_info
        self.ar_only = ar_only
        self.boost = boost

    @staticmethod
    def detect_multi_event_point(return_df, event_dict: dict):
        if return_df is None:
            return None, None
        cols = return_df.columns.tolist()
        new_event_dict = {}
        for k, v in event_dict.items():

            if isinstance(v, str):

                new_k = k + f"_{pd.to_datetime(v).strftime('%Y%m%d')}"
                new_event_dict[new_k] = v
                return_df[new_k] = return_df[k]
            elif isinstance(v, (list, tuple)):
                v = list(set(v))

                if len(v) >= 1:
                    # new_event_dict[k] = v
                    for dt in v:
                        new_k = k + f"_{pd.to_datetime(dt).strftime('%Y%m%d')}"
                        if k in cols:
                            return_df[new_k] = return_df[k]
                        else:
                            raise ValueError(f'{k} not found')
                        new_event_dict[new_k] = dt
                else:
                    raise ValueError(f'event list is empty for {k}')
            else:
                raise ValueError('got wrong event list type')
        return return_df, new_event_dict

    @property
    def arr(self):
        return self.ar

    @property
    def residual(self):
        if self.data is None:
            raise TypeError('return_df is not setup!')
        if self.event_dict is None:
            raise TypeError('event_dict is not setup!')
        if self.event_dict is None:
            raise TypeError('date and factors are not setup!')
        return self.cal_residual(self.data, self.event_dict, date=self.cols['date'],
                                 factors=self.cols['factors'], model=self.model,
                                 event_info=self.event_info, ar_only=self.ar_only, boost=self.boost)

    @cached_property
    def ar(self):
        """
            Cross-sectional aggregation
         the cross-sectional mean abnormal return for any period t


        :return:
        """
        return self.residual.mean(axis=1)

    @cached_property
    def std_ar(self):
        return self.ar.std()

    @cached_property
    def var_ar(self):
        return self.ar.var()

    @cached_property
    def t_stats(self):
        return self.car.squeeze() / np.sqrt(self.var_car)

    @cached_property
    def p_value(self):
        return 1.0 - t.cdf(abs(self.t_stats), event_tuple(*self.event_info).before_event_window - 1)

    @cached_property
    def var_car(self):
        """

        σ , = Lσ(AR)

        :return:
        """

        return [(i * var) for i, var in enumerate([self.var_ar] * self.ar.shape[0], 1)]

    #
    @cached_property
    def car(self):
        """

        The cumulative average residual method (CAR) uses as the abnormal performance measure
        the sum of each month’s average abnormal performance

        :return:
        """
        return pd.DataFrame(self.ar).cumsum()

    @cached_property
    def result(self):
        p = self.p_value
        t_statistic = self.t_stats.tolist()
        ar = self.ar.tolist()
        var_ar = [self.var_ar] * len(ar)
        car = self.car.squeeze()
        var_car = self.var_car
        return pd.DataFrame([ar, var_ar, car, var_car, t_statistic, p], columns=car.index,
                            index=['ar', 'var_ar', 'car', 'var_car', 't_stats', 'p_values']).T


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
# aar_df = EventStudy.cal_aar(es.ar)
# car_series = EventStudy.cal_car(es.ar)

# mean_AAR = es.ar
# var_AAR = es.var_residual
# var_matrix = pd.DataFrame(var_AAR)
# var_matrix = var_matrix.T
# var_AAR2 = sum(var_matrix.iloc[0]) / 15 ** 2
# Std_AAR = np.sqrt(var_AAR2)
if __name__ == '__main__':
    pass
