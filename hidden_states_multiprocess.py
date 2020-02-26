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
from multiprocessing import Process, Queue, Pool
import sqlite3
from time import sleep
import psutil
from itertools import product
warnings.simplefilter("ignore")

# TODO: make plots; fix bug with dates?

class stock_predictor():
    def __init__(self, params): 
    
    
        self.param_group_name = params.get('name', None)
        self.tickers = params.get('tickers', ['QLD'])
        
        if self.param_group_name is not None:
            params = self.get_params_from_db(self.param_group_name)
        elif self.param_group_name is None:
            self.param_group_name = params.get('name', namegenerator.gen())
        
        self.hold_length = params.get('hold_length', 1)
        self.period = params.get('period', '3y')
        self.k_features = params.get('k_features', 4)
        self.max_depth = params.get('max_depth', 5)
        self.scoring = params.get('scoring', 'neg_mean_squared_error')
        self.pattern = params.get('pattern', True)
        self.trade_other_states = params.get('trade_other_states', True)
        

        self.algo = params.get('algo', True)
        
        self.features = params.get('features', None)

        self.conn = sqlite3.connect('hidden_states.db')
        
        self.get_data()
        self.get_train_test()
        if self.features is None:
            
            if 'tree' in self.algo and 'original' not in self.algo:
                self.run_decision_tree()
            elif 'tree' in self.algo and 'original' in self.algo:
                self.run_decision_tree()
                self.features = list(set( self.features + ['return', 'range', 'close'] ))
                
            elif 'original' == self.algo:
                self.features = ['return', 'range', 'close']
                
            self.num_features = len(self.features)
            self.predict()
            self.get_trades()
            
            if len(self.all_trades) != 0:
                self.aggregate_results()
                self.store_results()
            
                
        elif self.features is not None:
            self.predict()
            #self.get_results()


    def get_params_from_db(self, name):
        
        conn = sqlite3.connect('hidden_states.db')
        sql = 'select * from hidden_states_test where name=="%s" limit 1' % name
        params = pd.read_sql(sql, conn)
        params[params.columns] = params[params.columns].apply(pd.to_numeric, errors='ignore')
        params = params.to_dict(orient='records')[0]
        params['tickers'] = self.tickers
        
        return params


    def get_data(self):
        all_historic_data = []
        
        for ticker in self.tickers:
            
            ticker_data = yfinance.Ticker(ticker)
            ticker_data = ticker_data.history(period=self.period, auto_adjust=False)
            
            ticker_data = get_ta(ticker_data, True, self.pattern)
            ticker_data = ticker_data.reset_index()
            ticker_data.columns = map(str.lower, ticker_data.columns)

            ticker_data["return"] = ticker_data["close"].pct_change()
            ticker_data["range"] = (ticker_data["high"]/ticker_data["low"])-1
            ticker_data = ticker_data.drop(columns=['dividends','stock splits'])

            ticker_data["ticker"] = ticker
            
            ticker_data.dropna(how="any", inplace=True)
            ticker_data = ticker_data.reset_index(drop=True)

            all_historic_data.append(ticker_data)
        
        self.history_df = pd.concat(all_historic_data)
        
        self.history_df = self.history_df.sort_values(by=['date'])
        self.history_df = self.history_df.reset_index(drop=True)
        
        
        

    def get_train_test(self):
        self.train = self.history_df[self.history_df['date']<'2018-12-31']
        self.test = self.history_df[ (self.history_df['date']>'2018-12-31') & (self.history_df['date']<'2019-12-31') ]
        self.test = self.test.loc[self.hold_length:]
        #print(self.train)
        
        #input()

    def run_decision_tree(self):
        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        sfs = SFS(clf, 
                k_features=self.k_features, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=-1,
                cv=5)
        while psutil.cpu_percent()>85:
            sleep(.5)

        self.train['target'] = self.train['close'].shift(-self.hold_length) / self.train['close'] - 1
        self.train = self.train.dropna()
        test_features = list(self.train.columns.drop(['date', 'ticker', 'return']))
        sfs = sfs.fit(self.train[test_features], self.train['return'])
        self.features = list(sfs.k_feature_names_)

        del self.train['target']


    def predict(self):
        model = mix.GaussianMixture(n_components=3, 
                                    covariance_type="full", 
                                    random_state=7,
                                    n_init=100)

        model.fit( self.train[  ['date'] + self.features].set_index("date") )

        self.test['state'] = model.predict(self.test[['date'] + self.features].set_index('date'))
        #print(self.test[ ['date', 'state'] + self.features ])
        #input()
        
    def get_spy(self):
        spy = yfinance.Ticker('SPY')
        spy = spy.history(period='10y')
        spy = spy.reset_index()
        spy.columns = map(str.lower, spy.columns)
        self.spy = spy


    def get_trades(self):

        self.get_spy()
        #print('before')
        #print(self.test)
        if self.trade_other_states:
            self.test.loc[self.test['state']==1, 'state'] = 2

        #print('after')
        #print(self.test)

        all_trades = []
        # make trades
        for ticker, possible_trades in self.test.groupby(by=['ticker']):
            buy_price = None
            #print(possible_trades)
            #input()
            
            for i in range(1,len(possible_trades)-1):
                yesterday = possible_trades.iloc[i-1]
                today = possible_trades.iloc[i]
                tomorrow = possible_trades.iloc[i+1]
                if today['state'] != yesterday['state'] and today['state']==2 and buy_price is None:
                    
                    #print('bought\n', today)
                    buy_price = float(tomorrow['open'])
                    buy_date = tomorrow['date']
                    spy_buy_price = self.spy[self.spy['date']==tomorrow['date']]['open'].values[0]
                elif today['state'] != yesterday['state'] and buy_price is not None:
                    sell_price = float(tomorrow['close'])
                    sell_date = tomorrow['date']
                    spy_sell_price = self.spy[self.spy['date']==tomorrow['date']]['close'].values[0]
                    #print('sold\n', today)
                    spy_change = spy_sell_price / spy_buy_price - 1
                    trade_percent_change = sell_price / buy_price - 1
                    abnormal_change = trade_percent_change - spy_change
                    all_trades.append([ticker, buy_date, sell_date,buy_price, sell_price, trade_percent_change, spy_change, abnormal_change])
                    buy_price = None
                    spy_buy_price = None
                    
                    
            # sell if currently held
            if buy_price is not None:
                sell_price = float(tomorrow['close'])
                sell_date = None
                spy_sell_price = self.spy[self.spy['date']==tomorrow['date']]['close'].values[0]

                spy_change = spy_sell_price / spy_buy_price - 1
                trade_percent_change = sell_price / buy_price - 1
                abnormal_change = trade_percent_change - spy_change

                all_trades.append([ ticker, buy_date, sell_date,buy_price, sell_price, trade_percent_change, spy_change, abnormal_change ])
                
        self.all_trades = all_trades

    def aggregate_results(self):
        self.all_trades = pd.DataFrame(self.all_trades, columns = ['ticker',  'buy_date', 'sell_date', 'buy_price', 'sell_price', 'trade_return', 'spy_return', 'abnormal_return'])
        print(self.all_trades)
        self.total_return = self.all_trades['trade_return'].sum()
        self.total_abnormal_return = self.all_trades['abnormal_return'].sum()
        self.num_trades = len(self.all_trades)
        self.accuracy = len(self.all_trades[self.all_trades['trade_return']>0]) / float(self.num_trades)

        self.total_result = [self.param_group_name, 
                             self.total_return,
                             self.total_abnormal_return,
                             self.num_trades, 
                             self.accuracy, 
                             self.all_trades['trade_return'].mean(), 
                             self.all_trades['trade_return'].median(), 
                             self.all_trades['trade_return'].std(),
                             self.all_trades['abnormal_return'].mean(),
                             self.all_trades['abnormal_return'].median(),
                             self.all_trades['abnormal_return'].std(),
                             ]
        self.total_result = pd.DataFrame([self.total_result], columns = ['name', 'total_return', 'total_abnormal_return', 'num_trades', 'accuracy', 'mean', 'median', 'stddev','abnormal_mean', 'abnormal_median', 'abnormal_stddev' ])

        
        print(self.total_result)

        self.individual_results = []
        for key, this_df in self.all_trades.groupby(by=['ticker']):
            num_trades = len(this_df)
            accuracy = len(this_df[this_df['trade_return']>0]) / float(num_trades)
            result = [self.param_group_name, key, this_df['trade_return'].sum(), num_trades, accuracy, this_df['trade_return'].mean(), this_df['trade_return'].median(), this_df['trade_return'].std()]
            self.individual_results.append(result)
        self.individual_results = pd.DataFrame(self.individual_results, columns = ['name', 'ticker', 'total_return', 'num_trades', 'accuracy', 'mean', 'median', 'stddev'])
        

    def store_results(self):
        
        results = [self.param_group_name, str(self.features), self.hold_length, 
                        self.period, self.pattern, self.trade_other_states, 
                        self.algo, self.num_features, self.max_depth, 
                        self.num_trades, self.total_return, self.accuracy]
        self.df = pd.DataFrame([results], columns = ['name', 'features', 'hold_length', 'period', 'pattern', 'trade_other_states', 'algo', 'num_features', 'max_depth', 'num_trades', 'total_return', 'accuracy'])

        self.all_trades['name'] = self.param_group_name

        self.all_trades.to_sql('hidden_states_models_trades', self.conn, if_exists='append')
        self.df.to_sql('hidden_states_models', self.conn, if_exists='append')
        self.total_result.to_sql('hidden_states_total_results', self.conn, if_exists='append')
        self.individual_results.to_sql('hidden_states_individual_results', self.conn, if_exists='append')
        

def run_model(params):
    stock_predictor(params)

if __name__ == '__main__':

    param_list = []
    #results = []
    
    periods = ['2y','3y', '4y','5y','6y']
    trade_other_states_list = [True,False]
    
    algos = ['tree', 'tree+original']
    hold_length = [1,2,3,4,5]
    pattern = [True,False]
    max_depth = list(range(2,8))
    k_features = list(range(3,11))

    inputs = product( periods, trade_other_states_list, algos, hold_length, pattern, max_depth, k_features )
    for period, trade_other_states, algo, hold_length, pattern, max_depth, k_features in list(inputs):
        param_list.append( {'tickers': ['QLD', 'SPY', 'QQQ', 'TQQQ', 'DJI'], 
                    'hold_length': hold_length, 
                    'period': period, 
                    'max_depth': max_depth, 
                    'pattern': pattern, 
                    'trade_other_states': trade_other_states, 
                    'k_features': k_features,
                    'algo': algo} )
    
    inputs = product( periods, trade_other_states_list )
    for period, trade_other_states in list(inputs):

        algo = 'original'
        hold_length = 1
        pattern = None
        max_depth = None
        k_features = None

        param_list.append( {'tickers': ['QLD', 'SPY', 'QQQ', 'TQQQ', 'DJI'], 
                            'hold_length': hold_length, 
                            'period': period, 
                            'max_depth': max_depth, 
                            'pattern': pattern, 
                            'trade_other_states': trade_other_states, 
                            'k_features': k_features,
                            'algo': algo} )

    

    
    print(len(param_list))                            
    p = Pool(16)
    p.map(run_model, param_list)
    #run_model(param_list[150])
    
