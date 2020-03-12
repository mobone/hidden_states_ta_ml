import yfinance
from ta_indicators import get_ta
import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_extraction import PrincipalComponentAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from mlxtend.evaluate import feature_importance_permutation
import matplotlib.pyplot as plt
from itertools import combinations
from random import shuffle
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count, Queue, Process
from time import sleep
import sqlite3
from joblib import dump, load
import namegenerator
import matplotlib
from scipy import stats
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
#from backtest_trader import Backtest, MyStrat
from strategy import setup_strategy
import sklearn.mixture as mix
import numpy as np
import seaborn as sns
import random
from random import randint
import warnings
from datetime import timedelta
warnings.simplefilter("ignore")
class model_check_error(Exception):
    pass

conn = sqlite3.connect('models.db')


class pipeline():
    def __init__(self, trains, test, features):
        self.name = namegenerator.gen()
        self.trains = trains
        self.test = test
        
        self.n_components = 3
        self.look_back = int(60)
        
        
        self.features = ['return'] + list(features)

        self.get_trained_pipelines()
        self.run_pipeline()
        self.get_state_summaries()
        #self.plot(self.test, show=True)

    def get_trained_pipelines(self):
        self.pipelines = []
        for train in self.trains:
            pipe_pca = make_pipeline(StandardScaler(),
                            PrincipalComponentAnalysis(n_components=self.n_components),
                            GMMHMM(n_components=self.n_components, covariance_type='full', n_iter=150, random_state=7),
                            )

            pipe_pca.fit(train[ self.features ])
            train['state'] = pipe_pca.predict(train[ self.features ])

            results = pd.DataFrame()
            for key, group in train.groupby(by='state'):
                results.loc[key, 'mean'] = group['return'].mean()
                results.loc[key, 'var'] = group['return'].std()
            results = results.sort_values(by='mean')
            results['new_state'] = list(range(self.n_components))

            if len(results[results['mean']<0])!=1:
                continue

            results = results.reset_index()
            results['name'] = namegenerator.gen()
            results['state_date'] = train['date'].head(1).values[0]
            results['end_date'] = train['date'].tail(1).values[0]
            #print(results)
            self.pipelines.append( [pipe_pca, results] )


    def run_pipeline(self):
        for i in range(self.look_back,len(self.test)+1):
            test = self.test.iloc[ i - self.look_back : i]
            today = test[-1:]

            max_score = -np.inf
            for pipeline, train_results in self.pipelines:
                try:
                    test_score = np.exp( pipeline.score( test[ self.features ]) / len(test) ) * 100
                    
                    if test_score>max_score:
                        
                        state = pipeline.predict( test[ self.features ] )[-1:][0]
                        
                        state = int(train_results[train_results['index']==state]['new_state'])
                        
                        
                        self.test.loc[today.index, 'state'] = state
                        self.test.loc[today.index, 'model_used'] = train_results['name'].values[0]
                        max_score = test_score
                except:
                    continue
        
        
        self.test = self.test.dropna(subset=['state'])
        self.num_models_used = len(self.test['model_used'].unique())
        
    def get_state_summaries(self):
        self.results = pd.DataFrame()
        for state, group in self.test.groupby(by='state'):
            self.results.loc[state, 'test_mean'] = group['return'].mean()
            self.results.loc[state, 'test_var'] = group['return'].std()

        for state, group in self.test.groupby(by='state'):
            self.results.loc[state, 'next_test_mean'] = group['next_return'].mean()
            self.results.loc[state, 'next_test_var'] = group['next_return'].std()
        
        self.results = self.results.sort_values(by='test_mean')
        self.results['num_models_used'] = self.num_models_used
        self.model_summary = self.results
        #print(self.results)

        
    def plot(self,df, show=False):
        df.loc[df['state']==0, 'color'] = 'firebrick'
        df.loc[df['state']==1, 'color'] = 'yellowgreen'
        df.loc[df['state']==2, 'color'] = 'forestgreen'
        df.loc[df['state']==3, 'color'] = 'darkslategray'

        df = df.dropna()
        df.plot.scatter(x='date',
                        y='close',
                        c='color',
                        )
                    
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        
        if show == True:
            plt.show()
        else: 
            plt.savefig('./plots/%s.png' % self.name)

        plt.close(fig)

def get_data(symbol, get_train_test=True):
        
        history = yfinance.Ticker(symbol).history(period='25y', auto_adjust=False).reset_index()
        if get_train_test:
            history = get_ta(history, volume=True, pattern=False)
        history.columns = map(str.lower, history.columns)
        
        history['date'] = pd.to_datetime(history['date'])
        history['return'] = history['close'].pct_change() * 100
        history = history.dropna()
        
        history.loc[history['high']<history['open'], 'high'] = history['open']+.01

        if get_train_test:
            history['next_return'] = history['return'].shift(-1)


            test = history.tail( 252*5 )

            start_train = 0
            trains = []
            train_length = 2
            while start_train+(252*train_length) < test.head(1).index:
                
                
                trains.append( history.iloc[start_train:start_train+(252*train_length)] )
                start_train = start_train + 252
                
            
            return trains, test
        else:
            return history


def get_backtest(name, symbol_1, symbol_2, df, short=False):
    df = df[ ['date', 'open', 'high', 'low', 'close', 'volume', 'state'] ]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'State']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    #print('starting')
    histories = {}
    filenames = []
    for symbol in [symbol_1, symbol_2]:
        history = get_data(symbol, get_train_test=False)
        history = history[ ['date', 'open', 'high', 'low', 'close', 'volume'] ]
        history.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        history['Adj Close'] = history['Close']
        history = history[ ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] ]
        history['Date'] = history['Date'].dt.strftime('%Y-%m-%d')
        history['Date'] = pd.to_datetime(history['Date'])
        history = history.set_index('Date')
        
        history['state'] = df['State']
        
        
        history = history.dropna()
        
        
        filename = "./trades/%s_%s.csv" % (name, symbol)
        history.to_csv(filename)
        filenames.append( [symbol, filename] )
        histories[symbol_1] = history
    
    
    df['Close'] = df['State']
    df['Low'] = 0.0
    df['Adj Close'] = df['Close']
    df = df[ ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] ]
    
    
    
    #print(df)
    filename = "./trades/%s_%s.csv" % (name, symbol_1+'_with_states')
    df.to_csv(filename)
    filenames.append( [symbol_1+'_with_states', filename] )
    #print('files saved')
    

    
    
    #input()
    backtest_results = setup_strategy(filenames, name)    
    #bt = basic_strat( histories, symbol_1, symbol_2 )
    #bt = Backtest(history, MyStrat, margin=1/2, cash=10000, commission=.0004, trade_on_close=1)

    #output = bt.run()
    #print(output)
    #bt.plot(plot_drawdown=True)
    import os
    for symbol, filename in filenames:
        try:
            os.remove(filename)
        except Exception as e:
            print(e)
    
    #print(backtest_results)



    

    return backtest_results

def pipeline_runner(trains, test, starting_features):
    good_features = ['stoch', 'roon', 'rsi', 'mom', 'beta', 'band', 'will']
    conn = sqlite3.connect('models_dynamic.db')
    while True:
        try:
                
            k_features = randint(1,4) * 3
            features = random.sample(starting_features, k_features)
            

            good = False
            for good_feature in good_features:
                if good_feature in str(features):
                    good = True
                    break
                
            if not good:
                continue
            print('testing', features)
            x = pipeline(trains.copy(), test.copy(), features)
            
            backtest_results = get_backtest(x.name, 'SSO', 'UPRO', x.test).T


            if backtest_results['sharpe_ratio'].values[0]<.7:
                raise model_check_error('sharpe ratio not great enough: %s' % float(backtest_results['sharpe_ratio']))
            #if backtest_results['cum_returns'].values[0] < 150:
                #raise model_check_error('returns not great enough: %s' % float(backtest_results['cum_returns']))
            backtest_results['name'] = x.name
            backtest_results['features'] = str(features)
            backtest_results['k_features'] = len(features)
            backtest_results.to_sql('backtests_dynamic', conn, if_exists='append')

            model_results = x.model_summary
            #print(model_results)
            model_results.loc[:, 'name'] = x.name
            model_results.loc[:, 'features'] = str(features)
            model_results.loc[:, 'k_features'] = len(features)
            model_results.to_sql('models_dynamic', conn, if_exists='append')
            print('')
            print(features)
            print(backtest_results)
            print(model_results)
            print('')

            x.plot(x.test, show=False)
        except Exception as e:
            print(e, features)


def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances').tail(60)
    
    starting_features = list(df['feature'].values)
    return starting_features

if __name__ == '__main__':
    trains, test = get_data('SPY')
    test_cols = list(test.columns.drop(['date','return', 'next_return']))
    starting_features = run_decision_tree(trains[0], test_cols)
    print(starting_features)

    for i in range(16):
        p = Process(target=pipeline_runner, args=(trains, test, starting_features,))
        p.start()

    while True:
        sleep(10)