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
from simple_trader_hmm import trader
conn = sqlite3.connect('tiny_pipeline.db')
class pipeline():
    def __init__(self, train, test, features, model_name = None):
        self.train = train
        self.test = test
        self.features = list(features)
        
        
        if model_name is None:
            self.get_model()
            if self.rename_passed == False:
                return
            self.predict_new()
            

            if not self.results[self.results['test_count']<25].empty:
                return
            if len(self.results.dropna())<3:
                return
            
            corr_1 = self.results[['train_mean','test_mean']].corr().values[0][1]
            corr_2 = self.results[['train_mean','test_next_mean']].corr().values[0][1]
            corr_3 = self.results[['test_mean','test_next_mean']].corr().values[0][1]

            # ensure at least one mean and var fits our needs
            best_state = self.results[self.results['test_mean']==self.results['test_mean'].max()]
            if float(best_state['test_mean'])<.25 or float(best_state['test_var'])>4.0:
                return

            if corr_1<0.6 or corr_2<0.6 or corr_3<0.6:
                return

            self.name = namegenerator.gen()
            self.results['corr_1'] = corr_1
            self.results['corr_2'] = corr_2
            self.results['corr_3'] = corr_3
            self.results['features'] = str(self.features)
            self.results['name'] = self.name
            

            
            self.results['safe_return'] = trader('QQQ', 'QLD', data = self.test).return_percentage
            self.results['moderate_return'] = trader('QQQ', 'TQQQ', data = self.test).return_percentage
            self.results['extreme_return'] = trader('QLD', 'TQQQ', data = self.test).return_percentage


            self.results.to_sql('models', conn, if_exists='append')
            #self.test[['date','close','state','state_name']].to_csv('%s.csv' % self.name)
            dump(self.pipe_pca, './models/%s.joblib' % self.name)
            self.plot()

            print(self.features)
            print(self.results)
        elif model_name is not None:
            self.name = model_name
            self.pipe_pca = load('./models/%s.joblib' % self.name)
            self.get_model()
            self.predict_new()
            self.results['extreme_return'] = trader('QLD', 'TQQQ', data = self.test).return_percentage
            print(self.test)
            
            self.plot(show=True)
            
            


    def apply_rename_states(self):
        
        for index, group in self.results.iterrows():
            self.test.loc[self.test['state']==index, 'state_name'] = group['state_name']
            self.test.loc[self.test['state']==index, 'state_num'] = group['state_num']
        print(self.results)
        # todo: renaming states is not working!!!
        print(self.test[ ['date','state'] ])
        self.test['state'] = self.test['state_num']
        print(self.test[ ['date','state','state_num','state_name','close'] ])
        

        
    def plot(self, show=False):
        self.test.plot.scatter(x='date',
                               y='close',
                               c='state',
                               colormap='viridis')
                    
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        file_name = self.name
        print('plotting', file_name)
        if show==False:
            plt.savefig('./plots_tiny_pipeline/%s.png'% file_name )
        else:
            plt.show()


    def get_model(self):
        
        self.pipe_pca = make_pipeline(StandardScaler(),
                         PrincipalComponentAnalysis(n_components=3),
                         GaussianHMM(n_components=3, covariance_type='full', random_state=7))

        self.pipe_pca.fit(self.train[ ['return'] + self.features ])
        model = self.pipe_pca.steps[2][1]
        
        results = []
        for i in range(3):
            result = [i, model.means_[i][0], np.diag(model.covars_[i])[0]]
            results.append(result)
            
        results = pd.DataFrame(results)
        results.columns = ['state', 'train_mean', 'train_var']
        self.results = results.set_index('state')

        self.get_renamed_states()
        

    def get_renamed_states(self):
        self.rename_passed = True
        self.results = self.results.sort_values(by=['train_var'])
        self.results['state_name'] = None
        self.results['state_num'] = None
        
        if len(self.results.loc[self.results['train_mean']<0])>1:
            self.rename_passed = False
            return 
        self.results.loc[self.results['train_mean']<0, 'state_name'] = 'sell'
        self.results.loc[self.results['train_mean']<0, 'state_num'] = 0
        
        # select the remaining groups
        groups = self.results[pd.isnull(self.results).any(axis=1)].sort_values(by=['train_var'])
        
        # assing the one with the lowest variation
        
        self.results.loc[self.results.index == int(groups.head(1).index.values[0]), 'state_name'] = 'strong_buy'
        self.results.loc[self.results.index == int(groups.head(1).index.values[0]), 'state_num'] = 2
        
        
        self.results.loc[pd.isnull(self.results).any(axis=1), 'state_name'] = 'buy'
        self.results.loc[pd.isnull(self.results).any(axis=1), 'state_num'] = 1
        
        
        #print(self.results)
        self.results = self.results.dropna()
        #print(self.results)
        if len(self.results)<3:
            self.rename_passed = False
            
        

    def predict_new(self):
        self.test['state'] = self.pipe_pca.predict(self.test[ ['return'] + self.features ])

        self.apply_rename_states()

        for state, group in self.test.groupby(by='state'):
            self.results.loc[state, 'test_mean'] = group['return'].mean()
            self.results.loc[state, 'test_var'] = group['return'].std()
            self.results.loc[state, 'test_count'] = group['return'].count()
            self.results.loc[state, 'test_next_mean'] = group['next_return'].mean()
            self.results.loc[state, 'test_next_var'] = group['next_return'].std()
            
            

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

        train = history.head( int(num_rows * .75) )
        test = history.tail( int(num_rows *.25) )
        
        
        test_cols = train.columns.drop(['date','return', 'next_return'])

        return train, test, test_cols

def pipeline_runner(input_queue):
    while input_queue.qsize():
        train, test, features = input_queue.get()
        
        pipeline(train, test, features)
        


def queue_monitor(input_queue):
    while input_queue.qsize():
        start_size = input_queue.qsize()
        sleep(60)
        end_size = input_queue.qsize()
        diff = start_size - end_size
        print('====================')
        print('queue monitor')
        print(diff, round(end_size / diff, 2))
        print('====================')
        
    
    


if __name__ == '__main__':
    input_queue = Queue()

    train, test, test_cols = get_data('QLD')
    starting_features = run_decision_tree(train, test_cols)
    #features_from_previous()
    
    feature_combos = list(combinations(starting_features, 5))
    shuffle(feature_combos)
    
    feature_input_list = []
    for features in feature_combos:
        input_queue.put( [train, test, features] )
        
    
    for i in range(cpu_count()-1):
    #for i in range(1):
        
        p = Process(target=pipeline_runner, args=(input_queue,) )
        p.start()

    p = Process(target=queue_monitor, args=(input_queue, ))
    p.start()
    p.join()
    
    