import yfinance
from ta_indicators import get_ta
import pandas as pd
from hmmlearn.hmm import GaussianHMM
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

import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from backtest_trader import Backtest, MyStrat
import sklearn.mixture as mix

class model_check_error(Exception):
    pass

conn = sqlite3.connect('models.db')


class pipeline():
    def __init__(self, df):
        self.df = df
        
        self.n_components = 5
        self.look_back = 240 * 1
        self.update_frequency = 20
        self.features = ['return', 'mfi', 'ch_osc', 'pvt', 'bbands_lower_p']

    def run_pipeline(self):


        self.pipe_pca = make_pipeline(StandardScaler(),
                         PrincipalComponentAnalysis(n_components=self.n_components),
                         GaussianHMM(n_components=self.n_components, covariance_type='full', n_iter=150, random_state=7),
                         #mix.GaussianMixture(n_components=self.n_components, covariance_type='full', n_init=150, random_state=7)
                         )
        

        import numpy as np
        from sklearn.model_selection import TimeSeriesSplit
        
        X = self.df.reset_index()
        
        
        tscv = TimeSeriesSplit(max_train_size=self.look_back, n_splits=10)
        print(tscv)
        
        for train_index, test_index in tscv.split(X):
            
            train = X.loc[train_index]
            test = X
            
            self.pipe_pca.fit ( train[ self.features ] )

            print(train)

            test['state'] = self.pipe_pca.predict( test[ self.features ] )
            """
            
            for i in range(self.look_back,len(self.df)+1):
                test = self.df.iloc[ i - self.look_back : i + 1]
                today = test[-1:]

                state = self.pipe_pca.predict( test[ self.features ] )
                state = state[-1:]
                print(state)
                self.df.loc[today.index, 'state'] = state
                
            
            """
            df = test.dropna()
            print(df[['date', 'state']])
            self.plot(df)

        
        """
        results = []
        for i in range(self.n_components):
            
            results.append([ i, self.pipe_pca.steps[2][1].means_[i][0], np.diag(self.pipe_pca.steps[2][1].covariances_[i])[0] ])
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        
        
        result_df = result_df.sort_values(by='mean')

        result_df['renamed_state'] = [0,1,2]
        print(result_df)

        for index, group in result_df.iterrows():
            
            self.df.loc[self.df['state']==int(index), 'state_num'] = group['renamed_state']
        
        self.df['state'] = self.df['state_num']
        """

        #self.get_model_results()
        #print(self.results)
        #self.get_renamed_states()
        #print(self.results)
        #print(self.df)

    def plot(self,df):
        
        #self.df.loc[self.df['state_name']=='sell', 'color'] = 'firebrick'
        #self.df.loc[self.df['state_name']=='buy', 'color'] = 'yellowgreen'
        #self.df.loc[self.df['state_name']=='strong_buy', 'color'] = 'forestgreen'
        #self.df.loc[self.df['state']==0, 'color'] = 'firebrick'
        #self.df.loc[self.df['state']==1, 'color'] = 'yellowgreen'
        #self.df.loc[self.df['state']==2, 'color'] = 'forestgreen'
        df = df.dropna()
        df.plot.scatter(x='date',
                        y='close',
                        c='state',
                        #c='color',
                        colormap='viridis'
                        )
                    
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5*.75, 10.5*.75, forward=True)
        

        plt.show()

        plt.close(fig)

    def get_model_results(self):
        self.results = pd.DataFrame()
        for state, group in self.df.groupby(by='state'):
            self.results.loc[state, 'mean'] = group['return'].mean()
            self.results.loc[state, 'var'] = group['return'].std()
            

    def get_renamed_states(self):
        
        self.results = self.results.sort_values(by=['var'])
        self.results['state_name'] = None
        self.results['state_num'] = None
        #print('renaming states')
        #print(self.results)
        self.results.loc[self.results['mean']==self.results['mean'].min(), 'state_name'] = 'sell'
        self.results.loc[self.results['mean']==self.results['mean'].min(), 'state_num'] = 0

        # select the remaining groups
        groups = self.results[pd.isnull(self.results).any(axis=1)].sort_values(by=['var'])

        
        first_group = groups.iloc[0]
        second_group = groups.iloc[1]

        self.rename_step = None

        if float(first_group['mean']) > float(second_group['mean']) and float(first_group['var']) < float(second_group['var']):
            self.rename_step = 'first'
            self.results.loc[self.results.index == first_group.name, 'state_name'] = 'strong_buy'
            self.results.loc[self.results.index == first_group.name, 'state_num'] = 2
            self.results.loc[self.results.index == second_group.name, 'state_name'] = 'buy'
            self.results.loc[self.results.index == second_group.name, 'state_num'] = 1
        elif float(first_group['var']) < float(second_group['var']):
            self.rename_step = 'third'
            self.results.loc[self.results.index == first_group.name, 'state_name'] = 'strong_buy'
            self.results.loc[self.results.index == first_group.name, 'state_num'] = 2
            self.results.loc[self.results.index == second_group.name, 'state_name'] = 'buy'
            self.results.loc[self.results.index == second_group.name, 'state_num'] = 1
        else:
            raise model_check_error("failed at identifying states for rename")
        

        self.results['old_state'] = self.results.index

        for index, group in self.results.iterrows():
            self.df.loc[self.df['state']==int(group['old_state']), 'state_name'] = group['state_name']
            self.df.loc[self.df['state']==int(group['old_state']), 'state_num'] = group['state_num']
        
        self.df['state'] = self.df['state_num']


def get_backtest(symbol, df):
    


    history = get_data(symbol)
    history = history[ ['date', 'open', 'high', 'low', 'close', 'volume'] ]
    history.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    history = history.set_index('Date')

    states = df[ ['date','state'] ]
    states.columns = ['Date', 'State']
    states = states.set_index('Date')
    
    history['State'] = states
    
    history = history.dropna()
    print(history)
    
    bt = Backtest(history, MyStrat, margin=1/2, cash=10000, commission=.0004, trade_on_close=1)

    output = bt.run()
    print(output)
    bt.plot(plot_drawdown=True)
    return output



def get_data(symbol):
        
        history = yfinance.Ticker(symbol).history(period='10y', auto_adjust=False).reset_index()
        
        history = get_ta(history, volume=True, pattern=False)
        history.columns = map(str.lower, history.columns)
        history['return'] = history['close'].pct_change() * 100
        history = history.dropna()
        history['next_return'] = history['return'].shift(-1)

        return history

df = get_data('SPY')

x = pipeline(df)
x.run_pipeline()
x.plot()

get_backtest('QLD', x.df)