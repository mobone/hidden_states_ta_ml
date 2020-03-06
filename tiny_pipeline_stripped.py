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

class model_check_error(Exception):
    pass

conn = sqlite3.connect('tiny_pipeline.db')
class pipeline():
    def __init__(self, train, test, features, name):
        
        self.name = name
        self.train = train
        self.test = test
        self.features = list(features)


    def plot(self, df, show=False):
               
        df.loc[df['state_name']=='sell', 'color'] = 'firebrick'
        df.loc[df['state_name']=='buy', 'color'] = 'yellowgreen'
        df.loc[df['state_name']=='strong_buy', 'color'] = 'forestgreen'



        df.plot.scatter(x='date',
                        y='close',
                        #c='state_name',
                        c='color')
                    
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5*.75, 10.5*.75, forward=True)
        
        if show==False:
            file_name = self.name
            plt.savefig('./plots/%s.png'% file_name )
        else:
            plt.legend(fontsize='small')

            plt.show()

        plt.close(fig)

             
        

    def get_model(self):
        #print('building model', self.features)
        self.pipe_pca = make_pipeline(StandardScaler(),
                         PrincipalComponentAnalysis(n_components=3),
                         GaussianHMM(n_components=3, covariance_type='full', random_state=7))

        self.pipe_pca.fit (self.train[ ['return'] + self.features ] )        
        self.train['state'] = self.pipe_pca.predict( self.train[ ['return'] + self.features ] )
      
        

        individual_states = []
        for i in range(1,len(self.test)):
            
            test_slice = self.test.iloc[:i]
                        
            state = self.pipe_pca.predict( test_slice[ ['return'] + self.features ] )
            individual_states.append(int(state[-1:].copy()))
        
        
        
        individual_states = [0] + individual_states
        
        self.test['state'] = individual_states
        self.test = self.test.iloc[1:]
        self.test = self.test.tail(400)

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
        if len(negative_train_states)>1 or len(negative_test_states)>1: # or len(negative_next_test_states)>1:
            raise model_check_error('multiple negative means in states')

        if negative_train_states.empty or negative_test_states.empty: # or negative_next_test_states.empty:
            raise model_check_error('negative state does not exist')
        
        if int(negative_train_states.index.values[0]) != int(negative_test_states.index.values[0]): # or int(negative_test_states.index.values[0]) != int(negative_next_test_states.index.values[0]):
            raise model_check_error('negative state indexes do not match')

        
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
        elif float(second_group['train_mean']) > float(first_group['train_mean']) and float(second_group['train_var']) < float(first_group['train_var']):
            self.rename_step = 'second'
            self.results.loc[self.results.index == second_group.name, 'state_name'] = 'strong_buy'
            self.results.loc[self.results.index == second_group.name, 'state_num'] = 2
            self.results.loc[self.results.index == first_group.name, 'state_name'] = 'buy'
            self.results.loc[self.results.index == first_group.name, 'state_num'] = 1
        elif float(first_group['train_var']) < float(second_group['train_var']):
            self.rename_step = 'third'
            self.results.loc[self.results.index == first_group.name, 'state_name'] = 'strong_buy'
            self.results.loc[self.results.index == first_group.name, 'state_num'] = 2
            self.results.loc[self.results.index == second_group.name, 'state_name'] = 'buy'
            self.results.loc[self.results.index == second_group.name, 'state_num'] = 1
        elif float(second_group['train_var']) < float(first_group['train_var']):
            self.rename_step = 'fourth'
            self.results.loc[self.results.index == second_group.name, 'state_name'] = 'strong_buy'
            self.results.loc[self.results.index == second_group.name, 'state_num'] = 2
            self.results.loc[self.results.index == first_group.name, 'state_name'] = 'buy'
            self.results.loc[self.results.index == first_group.name, 'state_num'] = 1

        self.results['old_state'] = self.results.index
        
        #print(self.rename_step)
        #print(self.results)


    def apply_rename_states(self, df):
        
        for index, group in self.results.iterrows():
            df.loc[df['state']==int(group['old_state']), 'state_name'] = group['state_name']
            df.loc[df['state']==int(group['old_state']), 'state_num'] = group['state_num']
        
        #df['state'] = df['state_num']
        return df
        
        


    def get_model_results(self,df,name):
        
        for state, group in df.groupby(by='state'):
            self.results.loc[state, name+'_mean'] = group['return'].mean()
            self.results.loc[state, name+'_var'] = group['return'].std()
            if name == 'test':
                self.results.loc[state, name+'_count'] = group['return'].count()
                self.results.loc[state, name+'_next_mean'] = group['next_return'].mean()
                self.results.loc[state, name+'_next_var'] = group['next_return'].std()
        

        
        
        
            
            

def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['next_return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances').tail(50)
    
    starting_features = list(df['feature'].values)
    return starting_features

def get_data(symbol):
        
        history = yfinance.Ticker(symbol).history(period='7y').reset_index()
        
        history = get_ta(history, volume=True, pattern=False)
        history.columns = map(str.lower, history.columns)
        history['return'] = history['close'].pct_change() * 100
        history = history.dropna()
        history['next_return'] = history['return'].shift(-1)
        
        num_rows = len(history)

        train = history.head( int(num_rows * .70) )
        test = history.tail( int(num_rows *.30) )
        
        
        test_cols = train.columns.drop(['date','return', 'next_return'])

        return train, test, test_cols


def pipeline_runner(input_queue):
    train, test, test_cols = get_data('QQQ')
    while True:
        features = input_queue.get()
        try:
            name = namegenerator.gen()
            x = pipeline(train, test, features, name)
            x.get_model()
        except model_check_error:
            #print(e)
            continue
            

        safe_return = trader(x.test, 'QQQ', 'QLD').return_percentage
        moderate_return = trader(x.test, 'QQQ', 'TQQQ').return_percentage
        extreme_return = trader(x.test, 'QLD', 'TQQQ').return_percentage

        if safe_return<0 or moderate_return<0 or extreme_return<0:
            #print('model has negative return')
            continue
        
        corr_1 = x.results[['train_mean','test_mean']].corr().values[0][1]
        corr_2 = x.results[['train_mean','test_next_mean']].corr().values[0][1]
        corr_3 = x.results[['test_mean','test_next_mean']].corr().values[0][1]

        if corr_1<0 or corr_2<0 or corr_3<0:
                continue

        x.results['corr_1'] = corr_1
        x.results['corr_2'] = corr_2
        x.results['corr_3'] = corr_3

        
        safe_return = trader(x.test, 'QQQ', 'QLD').return_percentage
        moderate_return = trader(x.test, 'QQQ', 'TQQQ').return_percentage
        extreme_return = trader(x.test, 'QLD', 'TQQQ').return_percentage

        x.results['safe_return'] = safe_return
        x.results['moderate_return'] = moderate_return
        x.results['extreme_return'] = extreme_return

        if safe_return<0 or moderate_return<0 or extreme_return<0:
            continue

        safe_return = trader(x.test, 'QQQ', 'QLD', short_symbol = 'QID').return_percentage
        moderate_return = trader(x.test, 'QQQ', 'TQQQ', short_symbol = 'QID').return_percentage
        extreme_return = trader(x.test, 'QLD', 'TQQQ', short_symbol = 'QID').return_percentage

        x.results['safe_return_with_short'] = safe_return
        x.results['moderate_return_with_short'] = moderate_return
        x.results['extreme_return_with_short'] = extreme_return
                
        x.results['state_date'] = x.test['date'].head(1).values[0]
        x.results['end_date'] = x.test['date'].tail(1).values[0]
        x.results['name'] = name

        x.results['features'] = str(features)
        x.plot(x.test, show=False)
        
        #print(x.results)
        print('\t\tfouund model', features, name, safe_return, moderate_return, extreme_return)
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
    starting_features = run_decision_tree(train, test_cols)
    feature_combos = []
    #feature_combos.extend(list(combinations(starting_features, 3)))
    #feature_combos.extend(list(combinations(starting_features, 4)))
    feature_combos.extend(list(combinations(starting_features, 5)))
    #feature_combos.extend(list(combinations(starting_features, 6)))
    shuffle(feature_combos)
    """
    # use for individual testing
    while True:
        shuffle(feature_combos)
        try:
            name = namegenerator.gen()
            x = pipeline(train, test, feature_combos[0], name)
            x.get_model()
        except Exception as e:
            print(e)    
            continue
        x.plot(x.test, show=False)            

       
        safe_return = trader(x.test, 'QQQ', 'QLD').return_percentage
        moderate_return = trader(x.test, 'QQQ', 'TQQQ').return_percentage
        extreme_return = trader(x.test, 'QLD', 'TQQQ').return_percentage

        x.results['safe_return'] = safe_return
        x.results['moderate_return'] = moderate_return
        x.results['extreme_return'] = extreme_return

        if safe_return<0 or moderate_return<0 or extreme_return<0:
            continue

        safe_return = trader(x.test, 'QQQ', 'QLD', short_symbol = 'QID').return_percentage
        moderate_return = trader(x.test, 'QQQ', 'TQQQ', short_symbol = 'QID').return_percentage
        extreme_return = trader(x.test, 'QLD', 'TQQQ', short_symbol = 'QID').return_percentage

        x.results['safe_return_with_short'] = safe_return
        x.results['moderate_return_with_short'] = moderate_return
        x.results['extreme_return_with_short'] = extreme_return
                
        x.results['state_date'] = x.test['date'].head(1).values[0]
        x.results['end_date'] = x.test['date'].tail(1).values[0]
        x.results['name'] = name
        print(x.results)
        
    """
    
    
    input_queue = Queue()
    feature_input_list = []
    for features in feature_combos:
        input_queue.put( features )
        
    
    for i in range(cpu_count()):
    #for i in range(5):
        p = Process(target=pipeline_runner, args=(input_queue,) )
        p.start()
        

    p = Process(target=queue_monitor, args=(input_queue, ))
    p.start()
    p.join()
    
    
    
    