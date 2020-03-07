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
from simple_trader_stripped import trader
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from backtest_trader import Backtest, MyStrat, MyStratWithShort

class model_check_error(Exception):
    pass

conn = sqlite3.connect('tiny_pipeline_finding_features.db')
class pipeline():
    def __init__(self, train, test, features, name, window_length):
        
        self.name = name
        self.train = train
        self.test = test
        self.features = list(features)
        self.window_length = window_length


    def plot(self, df, show=False, window_length = ''):
               
        df.loc[df['state_name']=='sell', 'color'] = 'firebrick'
        df.loc[df['state_name']=='buy', 'color'] = 'yellowgreen'
        df.loc[df['state_name']=='strong_buy', 'color'] = 'forestgreen'
        df = df.dropna()
        df.plot.scatter(x='date',
                        y='close',
                        #c='state_name',
                        c='color')
                    
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5*.75, 10.5*.75, forward=True)
        
        if show==False:
            
            plt.savefig('./plots_new_features/%s.png'% (self.name + '_' +str(window_length)) )
        else:
            plt.legend(fontsize='small')

            plt.show()

        plt.close(fig)


    def get_model(self):
        #print('building model', self.features)

        # TODO: look into kNeighborsRegressor
        self.pipe_pca = make_pipeline(StandardScaler(),
                         PrincipalComponentAnalysis(n_components=3),
                         GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=7)
                         #GaussianMixture(n_components=3, covariance_type="full",  n_init=100, random_state=7)
                         #KNeighborsRegressor(n_neighbors=3)
                         
                         )

        self.pipe_pca.fit ( self.train[ ['return'] + self.features ] )        
        
        self.train['state'] = self.pipe_pca.predict( self.train[ ['return'] + self.features ] )
        #self.test['state'] = self.pipe_pca.predict( self.test[ ['return'] + self.features ] )
        
        
        """
        # this way is 100% consistent between iterated predictions and everything seen predictions
        # but it's missing the current day prediction
        scaler = StandardScaler()
        pca = PrincipalComponentAnalysis(n_components=3)
        model = GaussianMixture(n_components=3, covariance_type="full",  n_init=100, random_state=7)

        train_data = scaler.fit_transform(self.train[ ['return'] + self.features])
        pca.fit(train_data)
        train_data = pca.transform(train_data)
        train_states = model.fit_predict(train_data)
        
        self.train['state'] = train_states

        
        #self.train['state'] = self.pipe_pca.predict( self.train[ ['return'] + self.features ] )
        #self.test['seen_everything_states'] = self.pipe_pca.predict( self.test[ ['return'] + self.features ] )

        test_data = scaler.transform(self.test[ ['return'] + self.features ])
        test_data = pca.transform(test_data)
        states = model.predict(test_data)
        self.test['state'] = states
        """

        train_length = len(self.train)
        original_test_len = len(self.test)
        test_data = self.train.append(self.test)
        test_data = test_data[ ['date','return'] + self.features ]
        #test_data = self.test[ ['return'] + self.features ]
        
        
        individual_states = []
        for i in range(train_length,len(test_data)+1):
            test_slice = test_data.iloc[i-self.window_length:i]
            
            state = self.pipe_pca.predict(test_slice[ ['return'] + self.features ])
            
            individual_states.append( [test_slice['date'].tail(1).values[0], state[-1:][0]])
            
            
        individual_states = pd.DataFrame(individual_states).tail(original_test_len)
        individual_states.columns = ['date','state']
        
        
        
        self.test['state'] = individual_states['state'].values
        self.test = self.test.tail(252*4)

        
        
        self.train_score = np.exp( self.pipe_pca.score( self.train[ ['return'] + self.features ] ) / len(self.train) ) * 100

        #if self.train_score<1.5:
        #    raise model_check_error('train score not sufficent')

        self.test_score = np.exp( self.pipe_pca.score( self.test[ ['return'] + self.features ] ) / len(self.test) ) * 100

        self.results = pd.DataFrame()
        self.get_model_results(self.train, 'train')
        self.get_model_results(self.test, 'test')
        self.check_model()
        
        self.get_renamed_states()
        self.train = self.apply_rename_states(self.train)
        self.test = self.apply_rename_states(self.test)

        
        
    def check_model(self):
        
        self.results = self.results.sort_values(by=['train_mean'])
        
        # check that all states are used
        if len(self.results.dropna())<3:
            raise model_check_error('not all states used')
        
        negative_train_states = self.results[self.results['train_mean']<0]
        negative_test_states = self.results[self.results['test_mean']<0]
        #negative_next_test_states = self.results[self.results['test_next_mean']<0]
        if len(negative_train_states)>1 or len(negative_test_states)>1:# or len(negative_next_test_states)>1:
            raise model_check_error('multiple negative means in states')

        if negative_train_states.empty or negative_test_states.empty:# or negative_next_test_states.empty:
            raise model_check_error('negative state does not exist')
        
        if int(negative_train_states.index.values[0]) != int(negative_test_states.index.values[0]):# or int(negative_test_states.index.values[0]) != int(negative_next_test_states.index.values[0]):
            raise model_check_error('negative state indexes do not match')
        
        print(self.results)
        
    """
    def get_trained_model_results(self):

        
        model = self.pipe_pca.steps[2][1]
        
        results = []
        for i in range(3):
            result = [i, model.means_[i][0], np.diag(model.covars_[i])[0]]
            results.append(result)
            
        results = pd.DataFrame(results)
        results.columns = ['state', 'train_mean', 'train_var']
        self.results = results.set_index('state')
        """
        


    def get_renamed_states(self):
        self.rename_passed = True
        self.results = self.results.sort_values(by=['train_var'])
        self.results['state_name'] = None
        self.results['state_num'] = None
        #print('renaming states')
        #print(self.results)
        self.results.loc[self.results['train_mean']==self.results['train_mean'].min(), 'state_name'] = 'sell'
        self.results.loc[self.results['train_mean']==self.results['train_mean'].min(), 'state_num'] = 0

        # select the remaining groups
        groups = self.results[pd.isnull(self.results).any(axis=1)].sort_values(by=['train_var'])

        
        first_group = groups.iloc[0]
        second_group = groups.iloc[1]

        self.rename_step = None

        if float(first_group['train_mean']) > float(second_group['train_mean']) and float(first_group['train_var']) < float(second_group['train_var']):
            self.rename_step = 'first'
            self.results.loc[self.results.index == first_group.name, 'state_name'] = 'strong_buy'
            self.results.loc[self.results.index == first_group.name, 'state_num'] = 2
            self.results.loc[self.results.index == second_group.name, 'state_name'] = 'buy'
            self.results.loc[self.results.index == second_group.name, 'state_num'] = 1
        elif float(first_group['train_var']) < float(second_group['train_var']):
            self.rename_step = 'third'
            self.results.loc[self.results.index == first_group.name, 'state_name'] = 'strong_buy'
            self.results.loc[self.results.index == first_group.name, 'state_num'] = 2
            self.results.loc[self.results.index == second_group.name, 'state_name'] = 'buy'
            self.results.loc[self.results.index == second_group.name, 'state_num'] = 1
        else:
            raise model_check_error("failed at identifying states for rename")
        

        self.results['old_state'] = self.results.index
        
        #print(self.rename_step)
        #print(self.results)


    def apply_rename_states(self, df):
        #print('renaming states')
        #print(self.results)
        for index, group in self.results.iterrows():
            df.loc[df['state']==int(group['old_state']), 'state_name'] = group['state_name']
            df.loc[df['state']==int(group['old_state']), 'state_num'] = group['state_num']
        
        df['state'] = df['state_num']
        return df
        
        


    def get_model_results(self,df,name):
        for state, group in df.groupby(by='state'):
            self.results.loc[state, name+'_mean'] = group['return'].mean()
            self.results.loc[state, name+'_var'] = group['return'].std()
            if name == 'test':
                if group['return'].count()<50:
                    #print(group)
                    raise model_check_error('not enough states in count')
                self.results.loc[state, name+'_count'] = group['return'].count()
                self.results.loc[state, name+'_next_mean'] = group['next_return'].mean()
                self.results.loc[state, name+'_next_var'] = group['next_return'].std()
        

        
        
        
            
            

def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['next_return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    # select bottom half
    num = int(len(df)/2.0)
    df = df.sort_values(by='importances').tail(num)
    print(df)
    
    starting_features = list(df['feature'].values)
    return starting_features

def get_data(symbol):
        
        history = yfinance.Ticker(symbol).history(period='20y', auto_adjust=False).reset_index()
        
        history = get_ta(history, volume=True, pattern=False)
        history.columns = map(str.lower, history.columns)
        history['return'] = history['close'].pct_change() * 100
        history = history.dropna()
        history['next_return'] = history['return'].shift(-1)
        
        """
        num_rows = len(history)
        if amount == 'first':
            #history = history.tail( int(num_rows/3) )
            history = history.tail( 7500*1 )
            history = history.head( 7500 )
        elif amount == 'second':
            #history = history.tail( int(num_rows/3*1) )
            history = history.tail( int(2500*1.25) )
            history = history.head( 2500 )
        #elif amount == 'third':
            #history = history.tail( int(2500*1.5) )
            #history = history.head( 2500 )
        history = history.reset_index(drop=True)
        num_rows = len(history)
        #num_rows = len(history)
        train = history.head( int(num_rows * .75) )
        test = history.tail( int(num_rows *.25) )
        """
        train_start_date =  '2009-01-01'
        train_end_date =  '2012-12-31'
        train = history[ (history['date']>train_start_date) & (history['date']<train_end_date) ]

        test = history.tail((252*4) + 300)
        
        
        test_cols = train.columns.drop(['date','return', 'next_return'])

        return train, test, test_cols



def get_stock_data(symbol):
    x = yfinance.Ticker(symbol).history(period='10y', auto_adjust=False)
    x = x.reset_index()
    #x.columns = map(str.lower, x.columns)
    x['Date'] = pd.to_datetime(x['Date'])
    x = x.set_index('Date')
    return x

def get_backtest(name, symbol, df, with_short = False):
    
    
    history = get_stock_data(symbol)
    
    states = df[ ['date','state'] ]
    states.columns = ['Date', 'State']
    states = states.set_index('Date')
    history['State'] = states
    history = history.dropna()
    
    if with_short == False:
        strat = MyStrat
        filename = name + '_' + symbol + '.html'
    elif with_short == True:
        strat = MyStratWithShort
        filename = name + '_' + symbol + '_short.html'
    bt = Backtest(history, strat, margin=1/2, cash=10000, commission=.0004, trade_on_close=1)

    output = bt.run()
    output['With Short'] = with_short
    output['Symbol'] = symbol
    output['Name'] = name


    

    bt.plot(filename = './backtests_new_features/'+filename, open_browser = False, plot_drawdown=True)
    return output


def pipeline_runner(input_queue):
    
    while True:
        name = namegenerator.gen()
        if type(input_queue) is list:
            features = input_queue
        else:
            features = input_queue.get()
            

        for window_length in [100]:
            safe_return = 0
            train, test, test_cols = get_data('QQQ')
            
            try:
                
                x = pipeline(train, test, features, name, window_length)
                x.get_model()
            except Exception as e:
                
                #print(features, e)
                
                #print(test_amount)
                continue
                
            

            
            
            corr_1 = x.results[['train_mean','test_mean']].corr().values[0][1]
            corr_2 = x.results[['train_mean','test_next_mean']].corr().values[0][1]
            corr_3 = x.results[['test_mean','test_next_mean']].corr().values[0][1]

            if abs(corr_1)<0.10 or abs(corr_2)<0.10 or abs(corr_3)<0.10:
                continue

            x.results['corr_1'] = corr_1
            x.results['corr_2'] = corr_2
            x.results['corr_3'] = corr_3

            x.results['train_score'] = x.train_score
            x.results['test_score'] = x.test_score

            safe_results        =   get_backtest(name, 'QQQ', x.test)
            moderate_results    =   get_backtest(name, 'QLD', x.test)
            extreme_results     =   get_backtest(name, 'TQQQ', x.test)
            
            if safe_results['Return [%]']<75:
                continue

            backtest_results = pd.DataFrame([safe_results, moderate_results, extreme_results])
            del backtest_results['_strategy']
            backtest_results.to_sql('backtest_results', conn, if_exists = 'append', index=False)
            """
            try:
                if safe_results['Max. Drawdown [%]']>-30:
                    safe_results        =   get_backtest(name, 'QQQ', x.test, with_short=True)
                    moderate_results    =   get_backtest(name, 'QLD', x.test, with_short=True )
                    extreme_results     =   get_backtest(name, 'TQQQ', x.test, with_short=True)

                backtest_results = pd.DataFrame([safe_results, moderate_results, extreme_results])
                del backtest_results['_strategy']
                backtest_results.to_sql('backtest_results', conn, if_exists = 'append', index=False)
            except Exception as e:
                print('failed creating short trades', e)
            """
            x.results['buy_and_hold'] = safe_results['Buy & Hold Return [%]']
            x.results['safe_return'] = safe_results['Return [%]']
            x.results['safe_drawdown'] = safe_results['Max. Drawdown [%]']
            x.results['moderate_return'] = moderate_results['Return [%]']
            x.results['moderate_drawdown'] = moderate_results['Max. Drawdown [%]']
            x.results['extreme_return'] = extreme_results['Return [%]']
            x.results['extreme_drawdown'] = extreme_results['Max. Drawdown [%]']
            x.results['num_trades'] = safe_results['# Trades']
            x.results['share_ratio'] = safe_results['Sharpe Ratio']
            x.results['win_rate'] = safe_results['Win Rate [%]']
                    
            x.results['state_date'] = x.test['date'].head(1).values[0]
            x.results['end_date'] = x.test['date'].tail(1).values[0]
            x.results['name'] = name

            x.results['features'] = str(features)
            x.results['k_features'] = len(features)
            x.results['rename_step'] = x.rename_step
            x.results['window_length'] = window_length
            x.plot(x.test, show=False, window_length = window_length)
            
            #print(x.results)
            print('\t\tfouund model', window_length, features, name)
            print(backtest_results)
            x.results.to_sql('models', conn, if_exists='append')
            #print(input_queue.qsize())
        
    print('PROCESS EXITING')
        

        


def queue_monitor(input_queue):
    while input_queue.qsize():
        start_size = input_queue.qsize()
        sleep(60)
        end_size = input_queue.qsize()
        diff = start_size - end_size
        print('====================')
        print('queue monitor')
        if diff == 0:
            print('no change in queue')
            continue
        
        print(end_size, diff, round(end_size / diff, 2))
        print('====================')
        
    



if __name__ == '__main__':

    train, test, test_cols = get_data('QQQ')
    
    #starting_features = run_decision_tree(train, test_cols)
    starting_features = test_cols
    feature_combos = []
    print('creating features list')
    feature_combos.extend(list(combinations(starting_features, 3)))
    feature_combos.extend(list(combinations(starting_features, 4)))
    #feature_combos.extend(list(combinations(starting_features, 5)))
    #feature_combos.extend(list(combinations(starting_features, 6)))
    shuffle(feature_combos)
    print('features list created')
    
    
    input_queue = Queue()
    feature_input_list = []
    for features in feature_combos:
        #if 'stoch' not in str(features) or 'aroon' not in str(features) or 'beta' not in str(features) or 'mom' not in str(features):
            #continue
        input_queue.put( features )
    print('total qsize', input_queue.qsize())
    
    # LOOK INTO THESE FEATURES!
    #('mfi', 'ch_osc', 'pvt', 'bbands_lower_p')
    #pipeline_runner(features)
    
    #for i in range(cpu_count()/2):
    for i in range(8):
        p = Process(target=pipeline_runner, args=(input_queue,) )
        p.start()
    

    p = Process(target=queue_monitor, args=(input_queue, ))
    p.start()
    p.join()
    
    
    
    