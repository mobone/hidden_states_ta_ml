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
from redis_accuracy_class import model_generator
from time import sleep
import sqlite3
from multiprocessing import Pool, cpu_count, Process
from random import shuffle
import time

def get_backtest_dataset(symbol):
    history = yfinance.Ticker(symbol).history(period='7y', auto_adjust=False).reset_index()
    history.columns = map(str.lower, history.columns)
        
    history['date'] = pd.to_datetime(history['date'])
    history['return'] = history['close'].pct_change() * 100
    history = history.dropna()
    
    history.loc[history['high']<history['open'], 'high'] = history['open']+.01

    history.to_csv('./datasets/%s.csv' % symbol)


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

            
            test = history.tail( 252*6 )
            short_test = history.tail( int(252 * 3) )

            
            trains = []
            
            i = 0
            for train_length in range(1,3):
            
                start_train = 0
                while start_train+(252*train_length) < test.head(1).index:
                    set_name = 'strict_%s_%s_%s' % (int(start_train), int(train_length), i)
    
                    trains.append(  [set_name, history.iloc[start_train:start_train+(252*train_length)]] )
                    start_train = start_train + 252
                    i = i + 1

            """
            best_model_ranges = pd.read_csv('best_models.csv')
            i = 0 
            for _, row in best_model_ranges.iterrows():
                start = int(row['start'])
                end = int(row['start'] + row['length'])
                set_name = 'found_%s_%s_%s' % (int(row['start']), int(row['length']), i)
                trains.append( [set_name, history.iloc[start:end]] )
                i = i + 1
            """

            
            
            
            return trains, test, short_test
        else:
            return history




def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances').tail(16*2)
    print(df)
    
    starting_features = df['feature']
    top_starting_features = list(df.sort_values(by='importances').tail(8)['feature'].values)
    return starting_features, top_starting_features




def queue_creator(params):

    start_feature, test_length_name, svc_cutoff, scaler_name, name  = params
    start_feature = [start_feature]
    
    strat = AccuracyStrat

    features_available = list(pd.read_csv('./datasets/starting_features.csv')['feature'].values)
    
    conn =  sqlite3.connect('redis_results.db')
    q = Queue(is_async=False, connection=Redis( host='192.168.1.127' ))

    while len(start_feature)<16:
        jobs = []
        for new_feature in features_available:
            if new_feature in start_feature:
                continue
            this_features = start_feature + [new_feature]
            print('testing', name, this_features, svc_cutoff)
            

            #test, hmm_results, svc_results = model_generator(name, trains, test_data, this_features, scaler, svc_cutoff)        
            job_id = name + '__' + str(this_features)
            job = q.enqueue(model_generator, args=(name, test_length_name, this_features, svc_cutoff, scaler_name, ), job_timeout=3600,  result_ttl=86400 )
            jobs.append( (job, job_id) )
            

        
        best_job_results = None
        best_sharpe_ratio = 0
        best_features = None
        start_time = time.time()
        while True:
            results_df = []
            for job, job_id in jobs:
                features = job_id.split('__')[1]
                if job.result is None:
                    sharpe_ratio = None
                    
                else:    
                    sharpe_ratio = job.result[4]
                    if sharpe_ratio > best_sharpe_ratio:
                        best_sharpe_ratio = sharpe_ratio
                        best_features = features
                        best_job_results = [job.result[1], job.result[2], job.result[3]]
                results_df.append( [features, sharpe_ratio] )
            
            results_df = pd.DataFrame(results_df, columns = ['features', 'sharpe_ratio'])

            print(results_df)
            print(  len(results_df.dropna()) / float(len(results_df)) )
            
            
            if len(results_df[results_df['sharpe_ratio'].isnull()]):
                print('not complete. sleeping')
                sleep(5)
            else:
                break

            if (time.time() - start_time) > 600:
                break

        #best_features = results_df.sort_values(by=['sharpe_ratio']).tail(1)['features'].values[0]
        best_features = eval(best_features)
        print('found best features', best_features)
        
        backtest_results = best_job_results[2]
        if test_length_name == 'long':
            backtest_results['yearly_cum_returns'] = backtest_results['cum_returns'] / 3.0
        elif test_length_name == 'short':
            backtest_results['yearly_cum_returns'] = backtest_results['cum_returns'] / 6.0

        backtest_results['features'] = str(best_features)
        backtest_results['svc_cutoff'] = svc_cutoff
        backtest_results['test_length'] = test_length_name
        backtest_results['scaler'] = scaler_name
        backtest_results['name'] = name
        print(best_job_results[2])
        best_job_results[0].to_sql('hmm_results', conn, if_exists='append')
        best_job_results[1].to_sql('svc_results', conn, if_exists='append')
        best_job_results[2].to_sql('backtest_results', conn, if_exists='append')


        start_feature = best_features

    
if __name__ == '__main__':

    get_backtest_dataset('TQQQ')
    get_backtest_dataset('SQQQ')

    trains, long_test, short_test = get_data('SPY')

    for train_name, train_df in trains:
        train_df.to_csv('./datasets/%s.csv' % train_name)

    long_test.to_csv('./datasets/long_test.csv')
    short_test.to_csv('./datasets/short_test.csv')


    test_cols = list(long_test.columns.drop(['date','return', 'next_return']))
    starting_features, top_starting_features = run_decision_tree(trains[0][1], test_cols)

    starting_features.to_csv('./datasets/starting_features.csv')
    

    starting_features = list( starting_features.values )

    starting_features =  list(set( ['aroon_up', 'aroon_down', 'aroonosc','correl', 'mom', 'beta', 'rsi', 'bop', 
                                    'ultimate_oscillator', 'bbands_upper', 'bbands_middle', 'bbands_lower', 
                                    'bbands_upper_p', 'bbands_middle_p', 'bbands_lower_p', 'stochf_fastk', 'stochf_fastd', 'stochrsi_fastk', 'stochrsi_fastd' ] + starting_features ))


    scalers = ['standard', 'minmax']
    #test_lengths = [ ['short', short_test], ['long', long_test] ]
    test_lengths = [ 'short', 'long' ]
    #test_lengths = [ ['short', short_test] ]
    svc_cutoff = [1, .5, .25, .1]


    params_list = list(product( top_starting_features, test_lengths, svc_cutoff, scalers ))
    
    params_list_with_names = []
    for i in params_list:
        params_list_with_names.append( list(i) + [namegenerator.gen()] )

    shuffle(params_list_with_names)



    #for params in params_list_with_names:
        #queue_creator(params)
    #queue_creator(params_list_with_names[0])
    p = Pool(1)
    p.map(queue_creator, params_list_with_names)

    while True:
        sleep(10)
    #queue_parser()

        