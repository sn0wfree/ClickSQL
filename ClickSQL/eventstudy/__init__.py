# import pandas as pd
# import yfinance as yf
#
# ##没有安装yahoo finance的话可以这样安装：
# ## !pip install yfinance
#
# stock_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
#               'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC']
# ## AMZN: 亚马逊,  AAPL: 苹果, TSLA: 特斯拉, GE: 通用电气, GILD: 吉利德（做瑞德西韦那个公司）,
# ## BA: 波音, NFLX: 网飞, MS: 大摩,
# ## LNVGY: 联想 BABA: 阿里巴巴, LK: 瑞幸, JOBS: 前程无忧, CEO: 中海油, TSM: 台积电, JD: 京东,
# ## ^GSPC: SP500
# prices = pd.DataFrame()
# ## get data
# for stock in stock_list:
#     symbol = stock
#     tickerData = yf.Ticker(symbol)
#     price_data = tickerData.history(period='1d', start='2019-1-1', end='2020-4-1')['Close']
#     prices[stock] = price_data
# return_df = prices.pct_change().dropna()
#
# date = ['2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13',
#         '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23']
# date_df = pd.DataFrame(date, index=['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
#                                     'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD'], columns=['Date'])
# date_df.index.name = 'CompanyName'
#
# firm_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
#              'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD']
# return_df['RF'] = 0.0065
# return_df['Mkt_RF'] = return_df['^GSPC'] - return_df['RF']
# return_df.to_pickle('data.pkl')