import yfinance
from ta_indicators import get_ta
import pandas as pd
import namegenerator

from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlxtend.feature_extraction import PrincipalComponentAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from itertools import product
import numpy as np
from sklearn.svm import SVC
from itertools import combinations
from strategy import AccuracyStrat
from strategy import setup_strategy
import os
from rq import Queue
from redis import Redis
import glob
from time import sleep
def model_generator(name, test_length_name, features, svc_cutoff, scaler_name):
    
    pipelines = []
    
    if scaler_name == 'minmax':
        scaler = MinMaxScaler(feature_range = (0, 1))
    elif scaler_name == 'standard':
        scaler = StandardScaler()
    n_components = 3
    with_short = True
    
    look_back = int(126)

    files = glob.glob('./datasets/*.csv')
    trains = []
    for filename in files:
        if 'test' in filename or 'starting' in filename:
            continue
        trains.append( [filename, pd.read_csv(filename)] )

    if test_length_name == 'short':
        test = pd.read_csv('./datasets/short_test.csv')
    elif test_length_name == 'long':
        test = pd.read_csv('./datasets/long_test.csv')


    #print('starting!')
    
    def get_svc(train, test):
        svc_pipeline = make_pipeline( scaler ,
                            SVC(gamma='auto'),
                            )
        
        train['next_classification'] = 0
        train.loc[train['next_return']>svc_cutoff, 'next_classification'] = 1

        train.loc[train['next_return']< (-1*svc_cutoff), 'next_classification'] = -1
        
        svc_pipeline.fit(train[ features ], train['next_classification'])
        
        state = svc_pipeline.predict(test[ features ])[0]
        

        return state

    def get_trained_pipelines():
        
        for train_name, train in trains:
            
            try:
                pipe_pca = make_pipeline(scaler,
                                PrincipalComponentAnalysis(n_components=n_components),
                                GMMHMM(n_components=n_components, covariance_type='full', n_iter=150, random_state=7),
                                )
                #print(train)
                pipe_pca.fit(train[ features ])
                train['state'] = pipe_pca.predict(train[ features ])

                results = pd.DataFrame()
                for key, group in train.groupby(by='state'):
                    results.loc[key, 'mean'] = group['return'].mean()
                    results.loc[key, 'var'] = group['return'].std()
                results = results.sort_values(by='mean')
                results['new_state'] = list(range(n_components))

                if len(results[results['mean']<0])!=1:
                    continue

                results = results.reset_index()
                results['name'] = train_name
                
                pipelines.append( [pipe_pca, results, train] )
            except Exception as e:
                #print(train)
                #print('make trained pipelines exception', e)
                #sleep(10)
                pass

    

    def run_pipeline(test):
        rolling_pipeline = make_pipeline(scaler,
                            PrincipalComponentAnalysis(n_components=n_components),
                            GMMHMM(n_components=n_components, covariance_type='full', n_iter=150, random_state=7),
                            )
        
        
        
        for i in range(look_back,len(test)):
            
            this_test = test.iloc[ i - look_back : i]
            today = this_test[-1:]
            

            #print(today)
            max_score = -np.inf
            for pipeline, train_results, train in pipelines:
                try:
                    
                    test_score = np.exp( pipeline.score( this_test[ features ]) / len(this_test) ) * 100
                    
                    if test_score>max_score:
                        #print(test_score, max_score)
                        state = pipeline.predict( this_test[ features ] )[-1:][0]
                        
                        state = int(train_results[train_results['index']==state]['new_state'])
                        
                        
                        svc_state = get_svc(train, today)
                        #print(state, svc_state)

                        #self.test.loc[today.index, 'svc_state'] = svc_state
                        test.loc[today.index, 'state'] = state
                        test.loc[today.index, 'svc_state'] = svc_state
                        test.loc[today.index, 'model_used'] = train_results['name'].values[0]

                        max_score = test_score
                except Exception as e:
                    #print('this exception', e)
                    #sleep(10)
                    continue

        test = test.dropna(subset=['state'])
        models_used = str(test['model_used'].unique())
        num_models_used = len(test['model_used'].unique())
        

    def get_state_summaries():
        hmm_results = pd.DataFrame()
        svc_results = pd.DataFrame()
        """
        for state, group in test.groupby(by='state'):
            results.loc[state, 'hmm_test_mean'] = group['return'].mean()
            results.loc[state, 'hmm_test_var'] = group['return'].std()

        for state, group in test.groupby(by='state'):
            results.loc[state, 'hmm_next_test_mean'] = group['next_return'].mean()
            results.loc[state, 'hmm_next_test_var'] = group['next_return'].std()
        
        for state, group in test.groupby(by='state'):
            results.loc[state, 'hmm_count'] = group['next_return'].count()
        
        for state, group in test.groupby(by='svc_state'):
            results.loc[state, 'svc_test_mean'] = group['return'].mean()
            results.loc[state, 'svc_test_var'] = group['return'].std()

        for state, group in test.groupby(by='svc_state'):
            results.loc[state, 'svc_next_test_mean'] = group['next_return'].mean()
            results.loc[state, 'svc_next_test_var'] = group['next_return'].std()
        """
        for state, group in test.groupby(by='state'):
            hmm_results.loc[state, 'hmm_test_mean'] = group['return'].mean()
            hmm_results.loc[state, 'hmm_test_var'] = group['return'].std()
        for state, group in test.groupby(by= 'state'):
            hmm_results.loc[ state, 'hmm_next_test_mean'] = group['next_return'].mean()
            hmm_results.loc[ state, 'hmm_next_test_var'] = group['next_return'].std()
        for state, group in test.groupby(by= 'svc_state'):
            svc_results.loc[ state, 'svc_next_test_mean'] = group['next_return'].mean()
            svc_results.loc[ state, 'svc_next_test_var'] = group['next_return'].std()

        hmm_results = hmm_results.sort_values(by='hmm_next_test_mean')
        svc_results = svc_results.sort_values(by='svc_next_test_mean')
        try:
            print(hmm_results)
            print(svc_results)
        except:
            pass
        
        return hmm_results, svc_results

        
    def get_data(symbol):
            """
            history = yfinance.Ticker(symbol).history(period='7y', auto_adjust=False).reset_index()
            history.columns = map(str.lower, history.columns)
            
            history['date'] = pd.to_datetime(history['date'])
            history['return'] = history['close'].pct_change() * 100
            history = history.dropna()
            
            history.loc[history['high']<history['open'], 'high'] = history['open']+.01
            """
            history = pd.read_csv('./datasets/%s_test.csv' % symbol)
            
            return history


    def get_backtest(name, long_symbol, short_symbol, df, strat, with_short):
        df = df[ ['date', 'open', 'high', 'low', 'close', 'volume', 'state', 'svc_state' ] ]
        df = df.dropna()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'State', 'SVC State']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        #print('starting')

        #print('here')
        #print(df)
        histories = {}
        filenames = []
        for symbol in [long_symbol, short_symbol]:
            history = get_data(symbol)
            history = history[ ['date', 'open', 'high', 'low', 'close', 'volume'] ]
            history.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            history['Adj Close'] = history['Close']
            history = history[ ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] ]
            
            #history['Date'] = history['Date'].dt.strftime('%Y-%m-%d')
            history['Date'] = pd.to_datetime(history['Date'])
            history = history.set_index('Date')
            
            history['state'] = df['State']
            
            
            history = history.dropna()
            
            
            filename = "./trades/%s_%s.csv" % (name, symbol)
            history.to_csv(filename)
            filenames.append( [symbol, filename] )
            histories[long_symbol] = history
        
        
        df['Close'] = df['State']
        df['Low'] = -5
        df['Adj Close'] = df['Close']
        markov_states = df[ ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] ]

        #print(markov_states)
        
        
        
        #print(df)
        filename = "./trades/%s_%s.csv" % (name, long_symbol+'_with_states')
        markov_states.to_csv(filename)
        filenames.append( [long_symbol+'_with_states', filename] )

        df['Close'] = df['SVC State']
        df['Adj Close'] = df['Close']
        
        svc_states = df[ ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] ]
        #print(svc_states)

        filename = "./trades/%s_%s.csv" % (name, long_symbol+'_with_svc_states')
        svc_states.to_csv(filename)
        filenames.append( [long_symbol+'_with_svc_states', filename] )

        #print('files saved')
        

        
        
        #input()
        backtest_results = setup_strategy(filenames, name, strat, with_short)    
        #bt = basic_strat( histories, symbol_1, symbol_2 )
        #bt = Backtest(history, MyStrat, margin=1/2, cash=10000, commission=.0004, trade_on_close=1)

        #output = bt.run()
        #print(output)
        #bt.plot(plot_drawdown=True)
        
        
        for symbol, filename in filenames:
            try:
                if float(backtest_results.T['sharpe_ratio']) > .5 and 'states' in filename:
                    continue
                else:
                    os.remove(filename)
            except Exception as e:
                #print('file exception', e)
                pass
        
        #print(backtest_results)



        

        return backtest_results

    hmm_results, svc_results, backtest_results = None, None, None
    sharpe_ratio = -42
    try:
        get_trained_pipelines()
        
        run_pipeline(test)
        hmm_results, svc_results = get_state_summaries()

        backtest_results = get_backtest(name, 'TQQQ', 'SQQQ', test, AccuracyStrat, with_short).T
        try:
            print(backtest_results)
        except:
            pass
        sharpe_ratio = backtest_results['sharpe_ratio'].values[0]
    except Exception as e:
        #print('\texception', e)
        pass
    
    

    return test, hmm_results, svc_results, backtest_results, sharpe_ratio
        
