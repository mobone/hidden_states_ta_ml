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
from random import shuffle
import requests
import re
from requests_toolbelt.threaded import pool
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

warnings.simplefilter("ignore")



def get_market_cap(market_cap):
    
    if 'K' in market_cap:
        market_cap = float(market_cap[:-1])*1000
    elif 'M' in market_cap:
        market_cap = float(market_cap[:-1])*1000000
    elif 'B' in market_cap:
        market_cap = float(market_cap[:-1])*1000000000
    return market_cap

def get_industry_tickers(sector):
    
    finviz_url = 'https://finviz.com/screener.ashx?v=111&f=exch_nasd,ipodate_more5,sec_technology,sh_avgvol_o1000,sh_opt_optionshort&o=-marketcap'
    finviz_url = finviz_url + '&r=%s'
        
    
    finviz_page = requests.get(finviz_url % 1)
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

        for key, this_row in df.iterrows():
            cap = get_market_cap(this_row['Market Cap'])
            df.loc[key, 'Numerical Market Cap'] = cap
        
        total_etf_df.append(df)
    total_etf_df = pd.concat(total_etf_df)
    total_etf_df = total_etf_df.sort_values(by=['Numerical Market Cap'], ascending=False)
    print(total_etf_df)
    del total_etf_df['No.']

    return list(total_etf_df['Ticker'])
    
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
        self.trade_time = params.get('trade_time', 'open')
        #self.marketcap = params.get('marketcap', 'small')
        
        self.score = None
        """
        total_num_tickers = len(self.tickers)
        if self.marketcap == 'large':
            self.tickers = self.tickers[:int(total_num_tickers/2)]
        elif self.marketcap == 'small':
            self.tickers = self.tickers[int(total_num_tickers/2):]
        """

        #self.sector = params.get('sector', None)

        self.algo = params.get('algo', True)
        
        self.features = params.get('features', None)

        self.conn = sqlite3.connect('hidden_states.db')
        
        self.get_data()
        self.get_train_test()
        if self.features is None:
            #self.run_decision_tree()
            
            if 'tree' in self.algo and 'original' not in self.algo:
                self.run_decision_tree()
            elif 'tree' in self.algo and 'original' in self.algo:
                self.run_decision_tree()
                self.features = list(set( self.features + ['return', 'range', 'close'] ))
                
            elif 'original' == self.algo:
                self.features = ['return', 'range', 'close']
            
            """
            if 'return' in self.features:
                self.trade_time = 'open'
            """

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
            #print('getting data for', ticker)
            ticker_data = yfinance.Ticker(ticker)
            ticker_data = ticker_data.history(period=self.period, auto_adjust=False)
            
            ticker_data = get_ta(ticker_data, False, self.pattern)
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
        self.history_df.replace([np.inf, -np.inf], np.nan)

        self.history_df = self.history_df.dropna()
        
        self.history_df = self.history_df.replace([np.inf, -np.inf], np.nan)
        self.history_df = self.history_df.dropna()
        

    def get_train_test(self):
        self.train = self.history_df[self.history_df['date']<'2017-12-31']
        self.test = self.history_df[ (self.history_df['date']>'2017-12-31') & (self.history_df['date']<'2019-12-31') ]
        self.test = self.test.loc[self.hold_length:]
        #print(self.train)
        
        #input()

    def run_decision_tree(self):
        while psutil.cpu_percent()>85:
            sleep(.5)

        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        sfs = SFS(clf, 
                k_features=self.k_features, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=-1,
                cv=4)

        
        self.train['target'] = self.train['close'].shift(-self.hold_length) / self.train['close'] - 1
        
        self.train = self.train.dropna()
        #self.train.to_csv('data.csv')
        test_features = list(self.train.columns.drop(['date', 'ticker', 'target']))
        #sfs = sfs.fit(self.train[test_features], self.train['target'])
        

        sfs = sfs.fit(self.train[test_features], self.train['target'])
        
        

        #self.features = list(sfs.k_feature_names_)
        self.score = sfs.k_score_
        
        self.features = list(sfs.k_feature_names_)
        print('features', self.features)
        del self.train['target']


    def run_grid_search(self):
        while psutil.cpu_percent()>85:
            sleep(.5)

        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        sfs = SFS(clf, 
                k_features=4, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=-1,
                cv=4)


        pipe = Pipeline([('sfs', sfs), 
                        ('clf', clf)])

        param_grid = [
        {'sfs__k_features': range(2,11)}
        ]

        gs = GridSearchCV(estimator=pipe, 
                        param_grid=param_grid, 
                        scoring=self.scoring, 
                        n_jobs=-1, 
                        cv=4,
                        verbose=2,
                        iid=True,
                        refit=True)

        

        
        self.train['target'] = self.train['close'].shift(-self.hold_length) / self.train['close'] - 1
        
        self.train = self.train.dropna()
        #self.train.to_csv('data.csv')
        test_features = list(self.train.columns.drop(['date', 'ticker', 'target']))
        #sfs = sfs.fit(self.train[test_features], self.train['target'])
        

        gs = gs.fit(self.train[test_features], self.train['target'])
        
        

        #self.features = list(sfs.k_feature_names_)
        self.score = gs.best_score_
        
        self.features = list(gs.best_estimator_.steps[0][1].k_feature_names_)
        print('features', self.features)
        del self.train['target']


    def predict(self):
        # TODO: generate model every day
        model = mix.GaussianMixture(n_components=3, 
                                    covariance_type="full", 
                                    random_state=7,
                                    n_init=100)

        model.fit( self.train[  ['date'] + self.features].set_index("date") )

        self.test['state'] = model.predict(self.test[['date'] + self.features].set_index('date'))
        #print(self.test[ ['date', 'state'] + self.features ])
        #input()
    """
    def get_spy(self):
        spy = yfinance.Ticker('QQQ')
        spy = spy.history(period='10y')
        spy = spy.reset_index()
        spy.columns = map(str.lower, spy.columns)
        self.spy = spy
    """

    def get_trades(self):

        #self.get_spy()
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
                    if self.trade_time == 'open':
                        buy_price = float(today['open'])
                        buy_date = today['date']
                    
                        #spy_buy_price = self.spy[self.spy['date']==tomorrow['date']]['open'].values[0]
                    elif self.trade_time == 'close':
                        buy_price = float(today['close'])
                        buy_date = today['date']
                    
                        #spy_buy_price = self.spy[self.spy['date']==today['date']]['close'].values[0]
                elif today['state'] != yesterday['state'] and buy_price is not None:
                    if self.trade_time == 'open':
                        sell_price = float(today['open'])
                        sell_date = today['date']
                        #spy_sell_price = self.spy[self.spy['date']==tomorrow['date']]['close'].values[0]

                    elif self.trade_time == 'close':
                        sell_price = float(today['close'])
                        sell_date = today['date']

                    
                        #spy_sell_price = self.spy[self.spy['date']==today['date']]['close'].values[0]
                    #print('sold\n', today)
                    #spy_change = spy_sell_price / spy_buy_price - 1
                    trade_percent_change = sell_price / buy_price - 1
                    #abnormal_change = trade_percent_change - spy_change
                    #all_trades.append([ticker, buy_date, sell_date,buy_price, sell_price, trade_percent_change, spy_change, abnormal_change])
                    all_trades.append([ticker, buy_date, sell_date,buy_price, sell_price, trade_percent_change, self.trade_time])
                    buy_price = None
                    #spy_buy_price = None
                    
                    
            # sell if currently held
            if buy_price is not None:
                sell_price = float(tomorrow['close'])
                sell_date = None
                #spy_sell_price = self.spy[self.spy['date']==tomorrow['date']]['close'].values[0]

                #spy_change = spy_sell_price / spy_buy_price - 1
                trade_percent_change = sell_price / buy_price - 1
                #abnormal_change = trade_percent_change - spy_change

                all_trades.append([ ticker, buy_date, sell_date,buy_price, sell_price, trade_percent_change, self.trade_time ])
                
        self.all_trades = all_trades

    def aggregate_results(self):
        #self.all_trades = pd.DataFrame(self.all_trades, columns = ['ticker',  'buy_date', 'sell_date', 'buy_price', 'sell_price', 'trade_return', 'spy_return', 'abnormal_return'])
        self.all_trades = pd.DataFrame(self.all_trades, columns = ['ticker',  'buy_date', 'sell_date', 'buy_price', 'sell_price', 'trade_return' , 'trade_time'])
        #print(self.all_trades)
        self.total_return = self.all_trades['trade_return'].sum()
        #self.total_abnormal_return = self.all_trades['abnormal_return'].sum()
        self.num_trades = len(self.all_trades)
        self.accuracy = len(self.all_trades[self.all_trades['trade_return']>0]) / float(self.num_trades)

        self.total_result = [self.param_group_name, 
                             self.total_return,
                             self.num_trades, 
                             self.accuracy, 
                             self.all_trades['trade_return'].mean(), 
                             self.all_trades['trade_return'].median(), 
                             self.all_trades['trade_return'].std(),
                             ]
        #self.total_result = pd.DataFrame([self.total_result], columns = ['name', 'total_return', 'total_abnormal_return', 'num_trades', 'accuracy', 'mean', 'median', 'stddev','abnormal_mean', 'abnormal_median', 'abnormal_stddev' ])
        self.total_result = pd.DataFrame([self.total_result], columns = ['name', 'total_return', 'num_trades', 'accuracy', 'mean', 'median', 'stddev'])

        
        

        self.individual_results = []
        for key, this_df in self.all_trades.groupby(by=['ticker']):
            num_trades = len(this_df)
            accuracy = len(this_df[this_df['trade_return']>0]) / float(num_trades)
            result = [self.param_group_name, key, this_df['trade_return'].sum(), num_trades, accuracy, this_df['trade_return'].mean(), this_df['trade_return'].median(), this_df['trade_return'].std()]
            self.individual_results.append(result)
        self.individual_results = pd.DataFrame(self.individual_results, columns = ['name', 'ticker', 'total_return', 'num_trades', 'accuracy', 'mean', 'median', 'stddev'])
        

    def store_results(self):
        
        if self.total_result['mean'].values[0]<0:
            return
        print(self.total_result)
        model_params = [str(self.features), self.hold_length, 
                        self.period, self.pattern, self.trade_other_states, 
                        self.algo, self.num_features, self.max_depth, self.scoring, self.score]
        self.df = pd.DataFrame([model_params], columns = ['features', 'hold_length', 'period', 'pattern', 'trade_other_states', 'algo', 'num_features', 'max_depth', 'scoring',  'score'])

        self.all_trades['name'] = self.param_group_name
        self.df['trade_time'] = self.trade_time
        #self.df['marketcap'] = self.marketcap
        self.df = pd.concat( [self.total_result, self.df,], axis=1)

        self.all_trades.to_sql('forex_trades', self.conn, if_exists='append', index=False)
        self.df.to_sql('forex_models', self.conn, if_exists='append', index=False)
        #self.total_result.to_sql('hidden_states_total_results', self.conn, if_exists='append', index=False)
        self.individual_results.to_sql('forex_per_ticker_summary', self.conn, if_exists='append', index=False)
        

def run_model(params):
    stock_predictor(params)

if __name__ == '__main__':

    param_list = []
    #results = []
    """
    periods = ['4y','5y','6y','7y','8y','9y','10y']
    trade_other_states_list = [True,False]
    trade_time = ['open', 'close']
    
    algos = ['tree', 'tree+original', 'original']
    hold_length = [1,2,3,4,5,6]
    pattern = [True,False]
    max_depth = list(range(2,8))
    k_features = list(range(3,11))
    """

    periods = ['5y','6y','7y','8y','9y','10y','15y', 'max']
    trade_other_states_list = [True, False]
    trade_time = ['close', 'open']
    #marketcaps = ['large', 'small']

    algos = ['tree', 'tree+original']
    hold_length = [3,4,5,6,7,8,9,10]
    pattern = [True,False]
    max_depth = list(range(4,15))
    scoring = ['r2', 'neg_mean_squared_error']
    k_features = list(range(3,13))
    
    sector = 'technology'
    #tickers = ['EURUSD=X','USDJPY=X','GBPUSD=X','AUDUSD=X','USDCAD=X','USDCHF=X','USDCNY=X','USDHKD=X','EURGBP=X','USDKRW=X']
    tickers = ['GBPUSD=X','USDGBP=X']
    #tickers = get_industry_tickers(sector)
    print(tickers)
    
    inputs = list( product( periods, trade_other_states_list, algos, hold_length, pattern, max_depth, trade_time, scoring, k_features ) )
    shuffle(inputs)
    for period, trade_other_states, algo, hold_length, pattern, max_depth, trade_time, scoring, k_features in list(inputs):
        param_list.append( {'tickers': tickers, 
                    'hold_length': hold_length, 
                    'period': period, 
                    'max_depth': max_depth, 
                    'pattern': pattern, 
                    'trade_other_states': trade_other_states, 
                    'algo': algo,
                    'sector': sector, 
                    'trade_time': trade_time,
                    'scoring': scoring} )
    
    

    
    print(len(param_list))                            
    p = Pool(15)
    p.map(run_model, param_list)
    """
    for params in param_list:
        print(params)
        run_model(params)
        print('complete')
    """
        
    

# select trade_time, period, avg(num_trades), avg(mean), avg(accuracy), avg(abnormal_mean) from hidden_states_models_technology group by trade_time, period order by avg(abnormal_mean) desc