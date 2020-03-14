import yfinance
from ta_indicators import get_ta
import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
import os
from sklearn.svm import SVC
import time
from strategy import MyStrategy, MyStrategy_2, MyStrategy_3
warnings.simplefilter("ignore")
class model_check_error(Exception):
    pass

conn = sqlite3.connect('models_with_sfs_4_states.db')


class pipeline():
    
    def __init__(self, name, test_length, trains, test, features, model_type, with_svc, scaler):
        self.name = name +'_'+test_length+'_'+model_type
        self.model_type = model_type
        
        self.trains = trains
        self.test = test
        self.with_svc = with_svc
        self.scaler = scaler


        
        self.n_components = 4
        
        self.look_back = int(126)
        
       
        
        self.features = ['return'] + list(features)
        
        self.get_trained_pipelines()
            
                
        self.run_pipeline()
        self.get_state_summaries()
        #self.plot(self.test, show=True)

    def get_trained_pipelines(self):
        self.pipelines = []
        for train_name, train in self.trains:
            #print(train_name)
            #print(train)
            try:
                pipe_pca = make_pipeline(self.scaler,
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
                results['name'] = train_name
                #results['state_date'] = train['date'].head(1).values[0]
                #results['end_date'] = train['date'].tail(1).values[0]
                #print(results)
                self.pipelines.append( [pipe_pca, results, train] )
            except Exception as e:
                #print('make pipelines', e)
                pass



    def get_svc(self, train, test):
        svc_pipeline = make_pipeline(MinMaxScaler(feature_range = (0, 1)),
                            SVC(gamma='auto'),
                            )
        

        train['next_classification'] = np.where( train['next_return']>0, 1, 0)

        
        svc_pipeline.fit(train[ self.features ], train['next_classification'])

        #print(svc_pipeline.score( train[ self.features ], train['next_classification'] ))

        state = svc_pipeline.predict(test[ self.features ])[0]

        #print(state)
        #sleep(10)

        return state


        


    def run_pipeline(self):
        rolling_pipeline = make_pipeline(self.scaler,
                            PrincipalComponentAnalysis(n_components=self.n_components),
                            GMMHMM(n_components=self.n_components, covariance_type='full', n_iter=150, random_state=7),
                            )
        
        

        for i in range(self.look_back,len(self.test)+1):
            test = self.test.iloc[ i - self.look_back : i]
            today = test[-1:]

            
            
            if self.model_type == 'dynamic':
                max_score = -np.inf
                for pipeline, train_results, train in self.pipelines:
                    try:
                        test_score = np.exp( pipeline.score( test[ self.features ]) / len(test) ) * 100
                        
                        if test_score>max_score:
                            #print(test_score, max_score)
                            state = pipeline.predict( test[ self.features ] )[-1:][0]
                            
                            state = int(train_results[train_results['index']==state]['new_state'])
                            
                            if self.with_svc == True:
                                svc_state = self.get_svc(train, today)
                                #print(svc_state, state)
                                #sleep(1)
                                if svc_state != 1:
                                    state = 0
                                

                            #self.test.loc[today.index, 'svc_state'] = svc_state
                            self.test.loc[today.index, 'state'] = state
                            self.test.loc[today.index, 'model_used'] = train_results['name'].values[0]

                            max_score = test_score
                    except Exception as e:
                        #print(e)
                        #sleep(10)
                        continue

        self.test = self.test.dropna(subset=['state'])
        self.models_used = str(self.test['model_used'].unique())
        self.num_models_used = len(self.test['model_used'].unique())
        
    def get_state_summaries(self):
        self.results = pd.DataFrame()
        for state, group in self.test.groupby(by='state'):
            self.results.loc[state, 'test_mean'] = group['return'].mean()
            self.results.loc[state, 'test_var'] = group['return'].std()

        for state, group in self.test.groupby(by='state'):
            self.results.loc[state, 'next_test_mean'] = group['next_return'].mean()
            self.results.loc[state, 'next_test_var'] = group['next_return'].std()
        
        for state, group in self.test.groupby(by='state'):
            self.results.loc[state, 'count'] = group['next_return'].count()
            

        self.results = self.results.sort_values(by='test_mean')
        self.results['num_models_used'] = self.num_models_used
        self.results['models_used'] = self.models_used
        if len( self.results[self.results['count']<30] )>1:
            raise model_check_error('state not used enough')
        if len(self.results[self.results['test_mean']<0])>1:
            raise model_check_error('too many negative states')
        if len(self.results)<self.n_components:
            raise model_check_error('not all states used')
        if float(self.results['test_mean'].head(1))>0 or float(self.results['next_test_mean'].head(1))>0:
            raise model_check_error('negative state does not exist')
        self.model_summary = self.results
        print('summary results')

        print(self.results)
        sleep(10)

        
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
            plt.savefig('./all_models_plots/%s.png' % self.name)

        plt.close(fig)

def get_data(symbol, get_train_test=True):
        
        if get_train_test == True:
            history = yfinance.Ticker(symbol).history(period='25y', auto_adjust=False).reset_index()
        else:
            history = yfinance.Ticker(symbol).history(period='7y', auto_adjust=False).reset_index()
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
            short_test = history.tail( int(252 * 3) )

            
            trains = []
            
            train_start_date =  '2009-01-01'
            train_end_date =  '2012-12-31'
            main_train = history[ (history['date']>train_start_date) & (history['date']<train_end_date) ]
            trains.append( ['main_train', main_train] )

            i = 0
            for train_length in range(1,3):
            
                start_train = 0
                while start_train+(252*train_length) < test.head(1).index:
                    set_name = 'strict_%s_%s_%s' % (int(start_train), int(train_length), i)
    
                    trains.append(  [set_name, history.iloc[start_train:start_train+(252*train_length)]] )
                    start_train = start_train + 252
                    i = i + 1


            best_model_ranges = pd.read_csv('best_models.csv')
            i = 0 
            for _, row in best_model_ranges.iterrows():
                start = int(row['start'])
                end = int(row['start'] + row['length'])
                set_name = 'found_%s_%s_%s' % (int(row['start']), int(row['length']), i)
                trains.append( [set_name, history.iloc[start:end]] )
                i = i + 1

            
            
            
            return main_train, trains, test, short_test
        else:
            return history


def get_backtest(name, symbol_1, symbol_2, df, strat, short=False):
    df = df[ ['date', 'open', 'high', 'low', 'close', 'volume', 'state' ] ]
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
    backtest_results = setup_strategy(filenames, name, strat)    
    #bt = basic_strat( histories, symbol_1, symbol_2 )
    #bt = Backtest(history, MyStrat, margin=1/2, cash=10000, commission=.0004, trade_on_close=1)

    #output = bt.run()
    #print(output)
    #bt.plot(plot_drawdown=True)
    
    
    for symbol, filename in filenames:
        try:
            if float(backtest_results.T['sharpe_ratio']) > .7 and 'states' in filename:
                continue
            else:
                os.remove(filename)
        except Exception as e:
            print('file exception', e)
    
    #print(backtest_results)



    

    return backtest_results

def pipeline_runner(input_queue):
    model_type = 'dynamic'
    conn = sqlite3.connect('models_with_sfs_4_states.db')
    
    while input_queue.qsize():
        params, starting_features, trains = input_queue.get()
        
        features, with_svc, scaler_option, test_periods, strat_option = params
        features = [features]
        scaler_name, scaler = scaler_option
        test_length, test = test_periods
        strat_name, strat = strat_option

        print('got params')
        print(params)
        while len(features)<13:
                    
            #k_features = randint(2,5) * 3
            runs_results = []
            for new_feature in starting_features:
                if new_feature in features:
                    continue
                
                test_features = features + [new_feature]

                print(test_features)
                #features = test_features
            

                
                name = namegenerator.gen()
                
                try:
                    start_time = time.time()
                    print('\ttesting', test_length, with_svc, scaler_name, test_features)
                    x = pipeline(name, test_length, trains, test, test_features, model_type, with_svc, scaler)
                    
                    backtest_results = get_backtest(name, 'QLD', 'TQQQ', x.test, strat).T

                    end_time = time.time()
                    this_run_result = [test_features, float(backtest_results['sharpe_ratio']), end_time-start_time]
                    runs_results.append(this_run_result)

                    result_df = pd.DataFrame(runs_results, columns = ['features', 'score', 'time'])
                    print(result_df)
                    
                    if float(backtest_results['sharpe_ratio'])<.7:
                        raise model_check_error('sharpe ratio not great enough: %s' % float(backtest_results['sharpe_ratio']))
                        #print('sharp ratio not great enough')
                        #continue
                    

                    backtest_results['name'] = name
                    backtest_results['features'] = str(test_features)
                    backtest_results['k_features'] = len(test_features)
                    backtest_results['model_type'] = model_type
                    backtest_results['test_length'] = test_length
                    backtest_results['num_models_used'] = x.num_models_used
                    backtest_results['models_used'] = x.models_used
                    backtest_results['with_svc'] = with_svc
                    backtest_results['scaler_name'] = scaler_name
                    backtest_results['strat_name'] = strat_name
                    backtest_results.to_sql('backtests', conn, if_exists='append')

                    model_results = x.model_summary
                    #print(model_results)
                    model_results.loc[:, 'name'] = name
                    model_results.loc[:, 'features'] = str(test_features)
                    model_results.loc[:, 'k_features'] = len(test_features)
                    model_results.loc[:, 'model_type'] = model_type
                    model_results.loc[:, 'test_length'] = test_length
                    model_results.loc[:, 'num_models_used'] = x.num_models_used
                    model_results.loc[:, 'models_used'] = x.models_used
                    model_results.loc[:, 'with_svc'] = with_svc
                    model_results.loc[:, 'scaler_name'] = scaler_name
                    model_results.loc[:, 'strat_name'] = strat_name
                    model_results.to_sql('models', conn, if_exists='append')
                    print('')
                    print(test_features)
                    print(backtest_results)
                    print(model_results)
                    print('')

                    x.plot(x.test, show=False)
                except Exception as e:
                    print('\t\texception\t',e, test_length, with_svc, scaler_name, test_features)
                    #sleep(10)
            try:
                result_df = pd.DataFrame(runs_results, columns = ['features', 'score', 'time'])
                result_df = result_df.sort_values(by='score').tail(1)
                print(result_df)
                features = result_df['features'].values[0]
                print('now using', features)
            except Exception as e:
                print("NO BEST FEATURE FOUND", features)
                break







def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances').tail(35)
    print(df)
    
    starting_features = list(df['feature'].values)
    top_starting_features = list(df.sort_values(by='importances').tail(8)['feature'].values)
    return starting_features, top_starting_features

if __name__ == '__main__':
    main_train, trains, long_test, short_test = get_data('SPY')
    test_cols = list(long_test.columns.drop(['date','return', 'next_return']))
    
    starting_features, top_starting_features = run_decision_tree(trains[0][1], test_cols)
    starting_features =  list(set( ['aroon_up', 'aroon_down', 'aroonosc','correl', 'mom', 'beta', 'rsi', 'bop', 
                                    'ultimate_oscillator', 'bbands_upper', 'bbands_middle', 'bbands_lower', 
                                    'bbands_upper_p', 'bbands_middle_p', 'bbands_lower_p', 'stochf_fastk', 'stochf_fastd', 'stochrsi_fastk', 'stochrsi_fastd' ] + starting_features ))
    #print(starting_features)

    
    with_svc = [True, False]
    scalers = [ ['standard', StandardScaler()], ['minmax', MinMaxScaler(feature_range = (0, 1))] ]
    test_lengths = [ ['short', short_test], ['long', long_test] ]
    test_lengths = [ ['short', short_test] ]
    strats = [ ['strat_1', MyStrategy], ['strat_2', MyStrategy_2], ['strat_3', MyStrategy_3] ]
    #strats = [ ['strat_2', MyStrategy_2] ]
    
    
    params_list = list(product( top_starting_features, with_svc, scalers, test_lengths, strats ))
    shuffle(params_list)
    #print(params_list)
    
    input_queue = Queue()
    for i in params_list:
        input_queue.put( (i, starting_features, trains) )


    for i in range(16):
        p = Process(target=pipeline_runner, args=(input_queue,))
        p.start()
        

    while True:
        sleep(10)
