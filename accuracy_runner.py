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

            
            
            
            return trains, test, short_test
        else:
            return history




def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances').tail(8)
    print(df)
    
    starting_features = list(df['feature'].values)
    top_starting_features = list(df.sort_values(by='importances').tail(4)['feature'].values)
    return starting_features, top_starting_features



def get_backtest(name, long_symbol, short_symbol, df, strat, with_short):
    df = df[ ['date', 'open', 'high', 'low', 'close', 'volume', 'state', 'svc_state' ] ]
    df = df.dropna()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'State', 'SVC State']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    #print('starting')

    print('here')
    print(df)
    histories = {}
    filenames = []
    for symbol in [long_symbol, short_symbol]:
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
            print('file exception', e)
    
    #print(backtest_results)



    

    return backtest_results


def queue_creator(params):

    start_feature, test_lengths, svc_cutoff, name, features_available, trains  = params
    start_feature = [start_feature]
    #scaler_name, scaler = scalers
    test_length_name, test_data = test_lengths

    strat = AccuracyStrat
    with_short = True
    """
    print(start_feature)
    print(scalers)
    print(test_lengths)
    print(svc_cutoff)
    print(name)
    print(features_available)
    """
    q = Queue(connection=Redis( host='192.168.1.127' ))

    for new_feature in features_available:
        this_features = start_feature + [new_feature]
        print('testing', this_features)
        

        #test, hmm_results, svc_results = model_generator(name, trains, test_data, this_features, scaler, svc_cutoff)        
        job_id = name + '_' + str(this_features)
        q.enqueue(model_generator, args=(name, trains, test_data, this_features, svc_cutoff, ), job_id = job_id )
        #q.enqueue(model_generator_2, args=(name, ))

    

def queue_parser():
    q = Queue(connection=Redis( host='192.168.1.127' ))

    while True:

        # Retrieving jobs
        queued_job_ids = q.job_ids # Gets a list of job IDs from the queue
        queued_jobs = q.jobs # Gets a list of enqueued job instances
        print(queued_job_ids)
        print(queued_jobs)
        sleep(10)
        #print(test)
        #backtest_results = get_backtest(name, 'TQQQ', 'SQQQ', test, strat, with_short).T

        #print(backtest_results)
    



#if __name__ == '__main__':



trains, long_test, short_test = get_data('SPY')

test_cols = list(long_test.columns.drop(['date','return', 'next_return']))
starting_features, top_starting_features = run_decision_tree(trains[0][1], test_cols)


starting_features =  list(set( ['aroon_up', 'aroon_down', 'aroonosc','correl', 'mom', 'beta', 'rsi', 'bop', 
                                'ultimate_oscillator', 'bbands_upper', 'bbands_middle', 'bbands_lower', 
                                'bbands_upper_p', 'bbands_middle_p', 'bbands_lower_p', 'stochf_fastk', 'stochf_fastd', 'stochrsi_fastk', 'stochrsi_fastd' ] + starting_features ))


#scalers = [ ['standard', StandardScaler()], ['minmax', MinMaxScaler(feature_range = (0, 1))] ]
test_lengths = [ ['short', short_test], ['long', long_test] ]
test_lengths = [ ['short', short_test] ]
svc_cutoff = [.5,.25,.1]


params_list = list(product( top_starting_features, test_lengths, svc_cutoff ))
params_list_with_names = []
for i in params_list:
    params_list_with_names.append( list(i) + [namegenerator.gen(), starting_features, trains] )




#for params in params_list_with_names:
    #queue_creator(params)
queue_creator(params_list_with_names[0])

queue_parser()

    