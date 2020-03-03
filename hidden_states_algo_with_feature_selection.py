from utils import get_data, get_industry_tickers, walk_timeline, plot_results
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import mixture as mix
import numpy as np
import pandas as pd
from itertools import product
from random import shuffle
import yfinance
import sqlite3
import namegenerator
from multiprocessing import Process, Queue, Pool
from time import time
from utils import plot_results
from statistics import stdev
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from time import sleep
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.data import wine_data
from mlxtend.feature_extraction import PrincipalComponentAnalysis

warnings.simplefilter("ignore")
        # TODO: make class return results for multiple years, check for consistency 
        # TODO: strip down to bare minimum
        # TODO: combine trading with other momentum signals
        # TODO: use timedelta to define train dataset, and advance through the years, or # TODO: test different time scales
class model_generator():
    def __init__(self, params):
        #self.conn = sqlite3.connect('markov_models.db')
        

        #print(params)

        # iterate through these params
        #self.name = params.get('name', namegenerator.gen())
        self.scoring = params.get('scoring', 'max_error')
        self.k_features = params.get('k_features', 25)
        self.max_depth = params.get('max_depth', 4)
        self.ascending = params.get('ascending', False)
        self.period = params.get('period', 'Max')

        # new features, for pipeline
        self.cutoff_date = '2017-01-01'
        self.train_length = 365*1
        self.test_length = 365*3
        self.k_max_features = 15

        tickers = ['SPY']
        self.data = get_data(tickers,period=self.period,pattern=False)
        
        #self.run_generator()
        self.run_generator_pipeline()

    def run_generator_pipeline(self):
        self.target_variable = 'return'
        
        train, test = self.get_train_test_pipeline()

        self.test_features = list(train.columns.drop(['date','ticker', 'next_day_return', self.target_variable]))
        
        self.pre_feature_selection(train)
        self.test_features = self.best_features
        print(self.test_features)
        self.run_pipeline(train, test)

        test['state'] = self.best_pipeline.predict(test[self.test_features])
        
        
        
        #print(X_test_backup)
        #test.to_csv('test.csv')
        test.plot.scatter(x='date', y='close', c='state', colormap='viridis')
        plt.show()
        #plt.savefig('./plots/%s_3_states_mixture.png' % (i))

    def pre_feature_selection(self, train):
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel

        test = train.copy()
        #test['return'] = test['return'].shift(-1)
        #test = test.dropna()
        
        X = test[self.test_features]
        y = test[self.target_variable]
        
        clf = ExtraTreesRegressor(n_estimators=50, random_state=7)
        clf = clf.fit(X, y)
        best_features = pd.DataFrame([X.columns, clf.feature_importances_]).T
        best_features.columns = ['features', 'score']
        best_features = best_features.sort_values(by=['score'])
        self.best_features = list(best_features['features'].tail(30))
    

    def run_pipeline(self, train, test):

        X = train[self.test_features]
        y = train[self.target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 7)

        
        max_score = 0
        
        n_states = 3
        self.best_pipeline = None

        for i in range(2,25):
            pipe_pca = make_pipeline(StandardScaler(),
                            PrincipalComponentAnalysis(n_components=i),
                            #mix.GaussianMixture (n_components=3, random_state=7),
                            KNeighborsRegressor(n_neighbors=3),
                            )
            
            pipe_pca.fit(X_train, y_train)
            score = pipe_pca.score(X_test, y_test)
            future_score = pipe_pca.score(test[self.test_features], test[self.target_variable])
            
            if score>max_score:
                self.best_pipeline = pipe_pca
                max_score = score
                print(i)
                print('Transf. training accyracy: %.2f%%' % (pipe_pca.score(X_train, y_train)*100))
                print('Transf. test accyracy: %.2f%%' % (pipe_pca.score(X_test, y_test)*100))
                print('Future test accyracy: %.2f%%' % (future_score*100))



    
    def run_generator(self):
        
        self.cutoff_date = '2019-01-01'
        self.train_length = 365*1
        self.test_length = 365*1
        self.k_max_features = 15
        self.get_train_test()
        
        self.features = list(self.train.columns.drop(['date','ticker', 'return']))
        """
        # get pre feature selection
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel
        test = self.train.copy()
        test['return'] = self.train['return'].shift(-1)
        test = test.dropna()
        
        X = test[self.features]
        y = test['return']
        
        clf = ExtraTreesRegressor(n_estimators=50)
        clf = clf.fit(X, y)
        best_features = pd.DataFrame([X.columns, clf.feature_importances_]).T
        best_features.columns = ['features', 'score']
        best_features = best_features.sort_values(by=['score'])
        self.features = list(best_features['features'].tail(30))
        """

        print(self.features)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.train[self.features])
        
        train = scaler.transform(self.train[self.features].copy())
        train = pd.DataFrame(train, columns = self.features)
        
        train['date'] = self.train['date'].values
        train['ticker'] = self.train['ticker'].values
        train['return'] = self.train['return'].values

        test = scaler.transform(self.test[self.features].copy())
        test = pd.DataFrame(test, columns = self.features)
        test['date'] = self.test['date'].values
        test['ticker'] = self.test['ticker'].values
        test['return'] = self.test['return'].values
        self.train = train
        self.test = test
        
        # get features that work well with mixture
        self.get_features()
        print(self.features_df)

        for name, features_df in self.features_df.iterrows():

            self.name = name
            self.features = features_df['features']
            self.score = features_df['score']
            if len(self.features)<4:
                continue
            print(self.name, self.features, self.score)
            self.generate_markov_model()
            self.test_model()
            

            if len(self.test['state'].unique())<3:
                print('continue')
                continue
            self.get_results()
        

    def get_features(self):
        features_dict = {}
        kept_features = []
        #global_max_score = -np.inf
        while len(kept_features)<self.k_max_features:
            

            local_max_score = -np.inf

            test_features_list = []
            for feature in self.features:
                
                
                if 'cdl' in feature and abs(self.train[feature]).sum()<50:
                    #print('skipping', feature, abs(self.train[feature]).sum())
                    continue
                    
                
                if feature not in kept_features:
                    test_features_list.append( kept_features + [feature])
                
            #print(test_features_list)
            
            p = Pool(5)
            results = p.map(self.generate_markov_model, test_features_list)
            """
            input_size = len(test_features_list)
            input_queue = Queue()
            output_queue = Queue()
            for i in test_features_list:
                input_queue.put(i)
            processes = []
            for i in range(16):
                p = Process(target=self.generate_markov_model_with_queue, args = (input_queue, output_queue,) )
                p.start()
                processes.append(p)

            while output_queue.qsize()<input_size:
                print(output_queue.qsize(), input_size)
                sleep(1)
            results = []
            while output_queue.qsize():
                results.append(output_queue.get())
            """
            #results = []
            #for i in test_features_list:
            #    print(i)
            #    results.append(self.generate_markov_model(i))
            
            results = pd.DataFrame(results, columns=['features', 'score']).sort_values(by='score', ascending=self.ascending)
            results = results.dropna()
            #print(results)
            

            kept_features = list(results['features'].tail(1).values[0])
            best_local_score = float(results['score'].tail(1))
            print('---')
            print('best local features', kept_features, best_local_score)
            print('---')
            
            runs_name = namegenerator.gen()
            
            features_dict[runs_name] = {'features': kept_features, 'k_features': len(kept_features), 'score': best_local_score}
            #print('---')
            #print(pd.DataFrame.from_dict(features_dict).T)
            #print('---')
            """
            if best_local_score>global_max_score:
                best_found_features = kept_features
                global_max_score = best_local_score
                if len(best_found_features)>11:
                    break
            """
        
        self.features_df = pd.DataFrame.from_dict(features_dict).T
        #input()
        #self.features = kept_features
        #print('best global features found', self.features)

    def get_train_test_pipeline(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        train_start_date = pd.to_datetime(self.cutoff_date) - timedelta(days=self.train_length)
        train_end_date = pd.to_datetime(self.cutoff_date)

        test_end_date = pd.to_datetime(self.cutoff_date) + timedelta(days = self.test_length)
        #self.train = self.data[self.data['date']<self.this_test_dates['start_date']]
        train = self.data[ (self.data['date']>train_start_date) & (self.data['date']<train_end_date) ]
        test = self.data[ (self.data['date']>train_end_date) & (self.data['date']<test_end_date) ]
        
        return train, test

    def get_train_test(self):
        self.data['date'] = pd.to_datetime(self.data['date'])

        
        train_start_date = pd.to_datetime(self.cutoff_date) - timedelta(days=self.train_length)
        train_end_date = pd.to_datetime(self.cutoff_date)

        test_end_date = pd.to_datetime(self.cutoff_date) + timedelta(days = self.test_length)
        #self.train = self.data[self.data['date']<self.this_test_dates['start_date']]
        self.train = self.data[ (self.data['date']>train_start_date) & (self.data['date']<train_end_date) ]
        self.test = self.data[ (self.data['date']>train_end_date) & (self.data['date']<test_end_date) ]


    def generate_markov_model_with_queue(self, input_queue, output_queue):
        # use percent return and try to minimize variance and maximize return
        # TODO: generate model every day
        #print('hey')
        try:
            while input_queue.qsize()>0:
                features = input_queue.get()
                
                #print('testing feature', features)
                model = mix.GaussianMixture (n_components=3, 
                                            covariance_type="full", #tied
                                            random_state=7,
                                            n_init=60)
                #if 'date' in self.train.columns:
                #    self.train = self.train.set_index('date')
                
                model.fit( self.train[ ['return'] + features] )
                #score = model.lower_bound_
                score = model.score(self.test[ ['return'] + features ])
                #print(features, score)
                output_queue.put( [features, score] )
            #print('exiting')
        except Exception as e:
            print('EXCEPTION', e)
            


    def generate_markov_model(self, features=None, return_score = False):
        # use percent return and try to minimize variance and maximize return
        # TODO: generate model every day
        print('started', features)
        try:
            if features is None:
                features = self.features
            else:
                return_score = True
            #print('testing feature', features)
            model = mix.GaussianMixture (n_components=3, 
                                        covariance_type="full", #tied
                                        random_state=7,
                                        n_init=60)
            #if 'date' in self.train.columns:
            #    self.train = self.train.set_index('date')
            #print(self.train[ ['return'] + features])
            model.fit( self.train[ ['return'] + features] )
            score = model.lower_bound_
            #score = self.model.score(self.test[ ['return'] + features ])
            print('finished', features)
            if return_score:
                #return  [features, self.model.lower_bound_]
                return  [features, score]
            else:
                self.model = model
            
        except Exception as e:
            print(e)
            return [features, None]

        
    def test_model(self):
        
        self.test['state'] = self.model.predict(self.test[['return'] + self.features])
        print('model score', self.model.score( self.test[['return']+self.features]) )
        #print(self.model.score_samples( self.test[['return']+self.features] ))

        # replace states wth state names
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
        
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state').sort_values(by=['mean'])
        state_nums = list(result_df.index)
        print(result_df)

        print(self.test)
        self.test.loc[self.test['state']==state_nums[0], 'state'] = 'sell'
        self.test.loc[self.test['state']==state_nums[1], 'state'] = 'buy'
        self.test.loc[self.test['state']==state_nums[2], 'state'] = 'strong_buy'
        print(self.test)
        



    def get_results(self):
        """
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
        
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state').sort_values(by=['mean'])
        """
        #print('===')
        #print(result_df)
        self.test['next_change'] = self.test['close'].shift(-1) / self.test['close'] - 1
        
        for state in self.test['state'].unique():
            this_group = self.test.loc[self.test['state']==state, 'next_change']
            print(state, float(this_group.mean()), float(this_group.std()))
        print('===')
        try:
            plot_results(self.test, self.name)
        except Exception as e:
            print(e)



def generate_models(params):
    
    
    # todo: move get_data outside this method
    
    
    start_time = time()
    model_generator(params)
    end_time = time()
    print(params, round( end_time - start_time ,4 ))

if __name__ == '__main__':
    

    k_features = range(4,11)
    max_depth = range(3,11)
    
    k_features = [6]
    max_depth = [6]
    """
    params_list = list( product( k_features, max_depth, scoring, with_original, period ) )
    print('total number of models', len(params_list))
    
    shuffle(params_list)

    params_dicts = []
    for k_features, max_depth, scoring, with_original, period in params_list:
        params = {}
        params['k_features'] = k_features
        params['max_depth'] = max_depth
        params['scoring'] = scoring
        params['with_original'] = with_original
        params['period'] = period
        #params['without_bad_year'] = without_bad_year
        params_dicts.append( params )
    
    #p = Pool(15)
    #p.map(generate_models, params_dicts)
    """
    params = {}
    
    
    
    generate_models(params)
    #    input()    
    
