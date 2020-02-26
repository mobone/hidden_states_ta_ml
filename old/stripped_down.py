import pandas as pd
import yfinance
from sklearn import mixture as mix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend
from ta_indicators import get_ta
import warnings
import time
import namegenerator
from multiprocessing import Process, Queue, Pool, cpu_count
import sqlite3
from time import sleep
import psutil
import requests as r
import re
from requests_toolbelt.threaded import pool
from random import choices
from datetime import datetime
warnings.simplefilter("ignore")

# TODO: make plots; fix bug with dates?


def get_stock_tickers(asset_type='etf'):
    if asset_type == 'etf' or asset_type=='both':
        
        #finviz_url = 'https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund,ipodate_more1,sh_avgvol_o100,sh_opt_option&r=%s'
        finviz_url = 'https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund,ipodate_more5,sh_avgvol_o1000,sh_opt_optionshort&r=%s'
        
        
        finviz_page = r.get(finviz_url % 1)
        ticker_count = int(re.findall('Total: </b>[0-9]*', finviz_page.text)[0].split('>')[1])
        urls = []

        for ticker_i in range(1, ticker_count, 20):
            urls.append(finviz_url % ticker_i)
            #break

        p = pool.Pool.from_urls(urls)
        p.join_all()

        total_etf_df = []
        for response in p.responses():
            start = response.text.find('<table width="100%" cellpadding="3" cellspacing="1" border="0" bgcolor="#d3d3d3">')
            end = response.text.find('</table>',start)+10

            #tickers = re.findall(r'primary">[A-Z]*', response.text)
            df = pd.read_html(response.text[start:end])[0]
            df.columns = df.loc[0]
            df = df.drop([0])
            total_etf_df.append(df)
        total_etf_df = pd.concat(total_etf_df)
        del total_etf_df['No.']
        

    return list(total_etf_df['Ticker'])

class stock_predictor():
    def __init__(self, params): 
    
        self.tickers = params.get('ticker', ['SPY'])
        self.period = params.get('period', '3y')
        self.target_state = params.get('target_state', 2)
        self.features = ['date', 'return', 'range', 'close']
        self.hold_length = 1
        self.k_features = 4

        self.get_data()
        self.get_train_test()
        #self.find_best_features()
        self.features = ['date', 'open', 'plus_dm', 'stochf_fastk', 'ht_leadsine', 'cdl3starsinsouth', 'cdltasukigap', 'return']
        #self.features = set(self.features + ['date', 'intraday', 'return', 'range', 'close'])
        self.predict()
        self.get_results()
        self.trade_model()

    def get_data(self, with_decision_tree=False):
        all_data = []
        for ticker in self.tickers:
            
            ticker_data = yfinance.Ticker(ticker)
            
            ticker_data = ticker_data.history(period=self.period, auto_adjust=False)
            ticker_data = ticker_data.reset_index()
            ticker_data.columns = map(str.lower, ticker_data.columns)
            ticker_data["return"] = ticker_data["close"].pct_change()
            #ticker_data["return"] = ticker_data["close"] / ticker_data["open"] - 1
            #ticker_data['target'] = ticker_data["close"].shift(-self.hold_length) / ticker_data["close"] - 1
            ticker_data['intraday'] = ticker_data["close"] / ticker_data["open"] - 1
            ticker_data = ticker_data.drop(columns=['dividends','stock splits'])
            ticker_data = get_ta(ticker_data, pattern = True)
            ticker_data.columns = map(str.lower, ticker_data.columns)
            
            ticker_data["ticker"] = ticker
            
            ticker_data["range"] = (ticker_data["high"]/ticker_data["low"])-1
            
            ticker_data.dropna(how="any", inplace=True)

            all_data.append(ticker_data)
        

        self.history_df = pd.concat(all_data)
        self.history_df = self.history_df.sort_values(by=['date'])
        
        
        #self.history_df = pd.concat([self.all_historic_data, ticker_data])
        print(self.history_df)

    def find_best_features(self):
        clf = DecisionTreeRegressor(random_state=7)
        sfs = SFS(clf, 
                k_features=self.k_features, 
                forward=True, 
                floating=True, 
                scoring='r2',
                cv=3)
        test_features = list(self.train.columns.drop(['date', 'target', 'ticker', 'return']))
        sfs = sfs.fit(self.train[test_features], self.train['target'])

        self.features = list(sfs.k_feature_names_)
        print(self.features)

    def get_train_test(self):
        
        self.history_df = self.history_df.sort_values(by=['date'])
        #train_test_split = len(self.history_df[self.history_df['date']<'2018-12-31'])

        self.train = self.history_df[self.history_df['date']<'2018-12-31']
        self.test = self.history_df[self.history_df['date']>'2018-12-31']
        #self.test = self.history_df.loc[train_test_split+self.hold_length:]

    def predict(self):
        print(datetime.now(), 'Training')
        self.model = mix.GaussianMixture(n_components=3, 
                                    covariance_type="full", 
                                    n_init=100).fit( self.train[ set( self.features ) ].set_index("date") )
        print(datetime.now(), 'Done Training')
        # Predict the optimal sequence of internal hidden state
        self.test['state'] = self.model.predict(self.test[ set( self.features ) ].set_index('date'))
        print(self.test)

    def get_results(self):
        print("Means and vars of each hidden state")
        for i in range(self.model.n_components):
            print("{0}th hidden state".format(i))
            print("mean\n ", pd.DataFrame([self.model.means_[i]]))
            print("var\n", pd.DataFrame([np.diag(self.model.covariances_[i])]))
            print()

        for i in range(self.model.n_components):
            print(self.test[self.test['state']==i]['return'].describe())

        input()
    def trade_model(self):
        # make trades
        
        trades = []
        for ticker in self.test['ticker'].unique():
            buy_price = None
            test_df = self.test[self.test['ticker']==ticker]
            for i in range(2,len(test_df)-1):
                
                yesterday = test_df.iloc[i-1]
                today = test_df.iloc[i]
                tomorrow = test_df.iloc[i+1]
                
                
                if today['state']==self.target_state and yesterday['state']!=self.target_state and buy_price is None:
                    buy_price = float(tomorrow['open'])
                    buy_date = today['date']

                elif today['state'] == self.target_state and buy_price is not None:
                    sell_price = float(tomorrow['open'])
                    sell_date = tomorrow['date']
                    
                    trades.append([buy_date, sell_date,buy_price, sell_price, sell_price / buy_price - 1])
                    buy_price = None

            # sell if currently held
            if buy_price is not None:
                sell_price = float(today['close'])
                sell_date = None
                trades.append([buy_date, sell_date,buy_price, sell_price, sell_price / buy_price - 1])

        df = pd.DataFrame(trades, columns = ['buy_date', 'sell_date', 'buy_price', 'sell_price', 'return'])

        print(df['return'].describe())

tickers = get_stock_tickers()

params = {'ticker': choices( tickers , k=20)}
params = {'ticker': ['QLD']}
stock_predictor(params)