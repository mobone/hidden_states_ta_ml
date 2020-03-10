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

class model_check_error(Exception):
    pass

conn = sqlite3.connect('models.db')


class pipeline():
    def __init__(self, train, test, features):
        self.name = namegenerator.gen()
        self.train = train
        self.test = test
        
        self.n_components = 3
        self.look_back = 100
        
        
        self.features = ['return'] + list(features)

    def run_pipeline(self):
        self.pipe_pca = make_pipeline(StandardScaler(),
                         PrincipalComponentAnalysis(n_components=self.n_components),
                         GMMHMM(n_components=self.n_components, covariance_type='full', n_iter=150, random_state=7),
                         #GaussianHMM(n_components=self.n_components, covariance_type='full', n_iter=150, random_state=7),
                         #mix.GaussianMixture(n_components=self.n_components, covariance_type='full', n_init=150, random_state=7)
                         )
        
        self.train = self.train.reset_index()
        self.test = self.test.reset_index()
        self.pipe_pca.fit ( self.train[ self.features ] )

        self.train['state'] = self.pipe_pca.predict( self.train[ self.features ] )
        
        for i in range(self.look_back,len(self.test)+1):
            test = self.test.iloc[ i - self.look_back : i]
            today = test[-1:]

            state = self.pipe_pca.predict( test[ self.features ] )
            state = state[-1:]
            
            self.test.loc[today.index, 'state'] = state
            
        self.test = self.test.dropna()

        self.get_model_results()
        self.get_renamed_states()

        
    def plot_prob_dist(self, df, show=False):
        
        #gmm = self.pipe_pca.steps[2][1]
        fig, ax = plt.subplots(figsize=(10,7))

        # reshape observed returns
        x = df['return'].sort_values().values.reshape(-1,1)
        #x = self.test['return']
        

        agg_pdfs = []
        for key, group in df.groupby(by='state'):
            mu = group['return'].mean()
            sd = np.sqrt(group['return'].std())
            w = 1.0
            agg_pdfs.append( w * stats.norm.pdf(x, mu, sd)  )
        
        

        
        # sum density in case
        #summed_density = np.sum(np.array(agg_pdfs), axis=0)

        #ax.plot(x, summed_density, color='k')  

        # plot observed data distribution #and a single component gaussian fit aka norm fit
        sns.distplot(x, ax=ax, hist=True, hist_kws=dict(alpha=0.25),
                    kde=True, kde_kws=dict(lw=4, label='QQQ-kde'),
                    label='QQQ')

        # plot component gaussians  
        for i in range(len(agg_pdfs)): ax.plot(x, agg_pdfs[i], ls='--', label='Gaussian '+str(i));  

        
        plt.title('QQQ kde with component densities')
        plt.legend()
        if show==True:
            plt.show()
        else:
            plt.savefig('./plots/%s.png' %  ( self.name + '_pd' ) )
        

    def plot(self,df, show=False):
        
        #df['date_num'] = range(len(df))
        #df['date'] = pd.to_datetime(df['date'])
        #df = df.set_index
        #print(df['state'].values)
        #df.loc[df['state_name']=='sell', 'color'] = 'firebrick'
        #df.loc[df['state_name']=='buy', 'color'] = 'yellowgreen'
        #df.loc[df['state_name']=='strong buy', 'color'] = 'forestgreen'
        #df.loc[df['state']==0, 'color'] = 'firebrick'
        #df.loc[df['state']==1, 'color'] = 'yellowgreen'
        #df.loc[df['state']==2, 'color'] = 'forestgreen'
        #print(df[['state','color']])
        
        

        df = df.dropna()
        df.plot.scatter(x='date'
                        y='close',
                        c='state',
                        #c='color'
                        colormap = 'viridis'
                        )
                    
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5*.75, 10.5*.75, forward=True)
        
        if show == True:
            plt.show()
        else: 
            plt.savefig('./plots/%s.png' % self.name)

        plt.close(fig)
        

    def get_model_results(self):
        
        self.results = pd.DataFrame()
        for state, group in self.train.groupby(by='state'):
            self.results.loc[state, 'mean'] = group['return'].mean()
            self.results.loc[state, 'var'] = group['return'].std()

        for state, group in self.test.groupby(by='state'):
            self.results.loc[state, 'test_mean'] = group['return'].mean()
            self.results.loc[state, 'test_var'] = group['return'].std()

        self.results = self.results.sort_values(by='mean')
        
        #print(self.results)
        if float(self.results['mean'].head(1))>0 or float( self.results['test_mean'].head(1) )>0:
            print(self.results)
            raise model_check_error('negative states exception')
        

        
            

    def get_renamed_states(self):
        
        self.results = self.results.sort_values(by=['var'])
        self.results['state_name'] = None
        self.results['state_num'] = None
        #print('renaming states')
        #print(self.results)
        self.results.loc[self.results['mean']==self.results['mean'].min(), 'state_name'] = 'sell'
        self.results.loc[self.results['mean']==self.results['mean'].min(), 'state_num'] = 0

        

        self.results.loc[self.results['mean']==self.results['mean'].max(), 'state_name'] = 'strong buy'
        self.results.loc[self.results['mean']==self.results['mean'].max(), 'state_num'] = 2

        self.results.loc[self.results['state_name'].isnull(), 'state_name'] = 'buy'
        self.results.loc[self.results['state_num'].isnull(), 'state_num'] = 1
        #print(self.results)
        #input()

        """
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
        """

        self.results['old_state'] = self.results.index

        for index, group in self.results.iterrows():
            self.test.loc[self.test['state']==int(group['old_state']), 'state_name'] = group['state_name']
            self.test.loc[self.test['state']==int(group['old_state']), 'state_num'] = group['state_num']
        
        self.test['state'] = self.test['state_num']


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
    
    print(backtest_results)



    

    return backtest_results



def get_data(symbol, get_train_test=True):
        
        history = yfinance.Ticker(symbol).history(period='10y', auto_adjust=False).reset_index()
        if get_train_test:
            history = get_ta(history, volume=True, pattern=False)
        history.columns = map(str.lower, history.columns)
        history['return'] = history['close'].pct_change() * 100
        history = history.dropna()
        

        if get_train_test:
            history['next_return'] = history['return'].shift(-1)
            train_start_date =  '2009-01-01'
            train_end_date =  '2012-12-31'
            train = history[ (history['date']>train_start_date) & (history['date']<train_end_date) ]

            test = history.tail((252*3) + 300)

            #test_cols = train.columns.drop(['date','return', 'next_return'])

            return train, test
        else:
            return history

def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances').tail(60)
    
    starting_features = list(df['feature'].values)
    return starting_features

def run_ml(features):
    try:
        conn = sqlite3.connect('models.db')
        
        train, test = get_data('QQQ')
        
    
        x = pipeline(train, test, features)
        x.run_pipeline()
    
        #print(x.test[ ['date', 'state', 'close'] ])
        #print(x.test[ ['date', 'state', 'close'] ].dtypes)
        

        backtest_results = get_backtest(x.name, 'QLD', 'TQQQ', x.test).T
        #print(backtest_results)
        if backtest_results['cum_returns'].values[0] < 50:
            raise model_check_error('returns not great enough %s' % backtest_results['cum_returns'])
        backtest_results['name'] = x.name
        backtest_results['features'] = str(features)
        backtest_results.to_sql('backtests', conn, if_exists='append')

        model_results = x.results
        #print(model_results)
        model_results.loc[:, 'name'] = x.name
        model_results.loc[:, 'features'] = str(features)

        model_results.to_sql('models', conn, if_exists='append')

        x.plot(x.test, show=True)
        x.plot_prob_dist(x.test, show=True)
    except Exception as e:
        print(e)
        return



if __name__ == "__main__":

    train, test = get_data('QQQ')
    test_cols = train.columns.drop(['date','return', 'next_return'])
    starting_features = run_decision_tree(train, test_cols)

    print(starting_features)
    
    good_features = ['stoch', 'beta', 'mom', 'pvt', 'bands', 'rsi', 'roon']
    feature_combos = list(combinations(starting_features, 4))
    shuffle(feature_combos)
    kept_features_list = []
    for combo in feature_combos:
        for good_feature in good_features:

            if good_feature in str(combo):
                kept_features_list.append(combo)
                #print(combo)
                break


    print(len(kept_features_list))
    
    for f in kept_features_list:
        run_ml(f)
    #p = Pool(16)
    #p.map(run_ml, kept_features_list)



