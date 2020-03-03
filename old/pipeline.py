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
import matplotlib
from mlxtend.feature_extraction import PrincipalComponentAnalysis

warnings.simplefilter("ignore")
        # TODO: make class return results for multiple years, check for consistency 
        # TODO: strip down to bare minimum
        # TODO: combine trading with other momentum signals
        # TODO: use timedelta to define train dataset, and advance through the years, or # TODO: test different time scales
class model_generator():
    def __init__(self, params):
        
        self.conn = sqlite3.connect('markov_models.db')
        
        self.name = params.get('name', namegenerator.gen())
        self.period = params.get('period', 'Max')
        self.train_length = params.get('train_length', 3)
        self.test_length = params.get('test_length', 3)
        self.cutoff_date = '2017-01-01'
        self.train_length = 365 * self.train_length
        self.test_length = 365 * self.test_length
        self.target_variable = 'return'

        self.k_features = params.get('k_features', 25)
        self.k_neighbors = params.get('k_neighbors', 3)

        tickers = ['SPY']
        self.data = get_data(tickers,period=self.period,pattern=False)
        
        #self.run_generator()
        self.run_generator_pipeline()

    def store_results(self):
        output_dict = {}

        output_dict['name'] = self.name
        #output_dict['train_score'] = self.training_score
        output_dict['testing_score'] = self.testing_score
        output_dict['future_test_score'] = self.future_testing_score
        output_dict['corr'] = self.corr
        output_dict['k_features'] = self.k_features
        output_dict['k_neighbors'] = self.k_neighbors
        output_dict['pca_n_components'] = self.pca_n_components
        output_dict['train_length'] = self.train_length
        output_dict['test_length'] = self.test_length
        output_dict['features'] = str(self.best_features)
        print(output_dict)
        df = pd.DataFrame.from_dict(output_dict, orient='index').T
        df.to_sql('pipeline_models_SPY', self.conn, if_exists='append', index=False)


    def run_generator_pipeline(self):
        
        
        
        train, test = self.get_train_test_pipeline()

        self.pre_feature_selection(train)
        self.test_features = self.best_features
        #print(self.test_features)
        
        # run the pipeline with scaler, PCA, and kneighbors
        self.run_pipeline(train, test)

        
        # predict on test data
        if self.best_pipeline is None:
            return
        print(self.test_features)
        test['state'] = self.best_pipeline.predict(test[['return']+self.test_features])
        
        # plot
        if self.future_testing_score>.6:

            test.plot.scatter(x='date', y='close', c='state', colormap='viridis')
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5, forward=True)

            plt.show()
            plt.savefig('./plots/%s.png' % (self.name))

        test["next_1_day_return"] = test['close'].shift(-1) / test["close"] - 1
        test["next_5_day_return"] = test['close'].shift(-5) / test["close"] - 1
        #test["next_10_day_return"] = test['close'].shift(-10) / test["close"] - 1
        
        self.corr_1 = test[['state','next_1_day_return']].corr()['state']['next_1_day_return']
        self.corr_5 = test[['state','next_5_day_return']].corr()['state']['next_5_day_return']

        
        self.store_results()
        
        

    def get_train_test_pipeline(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        train_start_date = pd.to_datetime(self.cutoff_date) - timedelta(days=self.train_length)
        train_end_date = pd.to_datetime(self.cutoff_date)

        test_end_date = pd.to_datetime(self.cutoff_date) + timedelta(days = self.test_length)
        #self.train = self.data[self.data['date']<self.this_test_dates['start_date']]
        train = self.data[ (self.data['date']>train_start_date) & (self.data['date']<train_end_date) ]
        test = self.data[ (self.data['date']>train_end_date) & (self.data['date']<test_end_date) ]
        
        return train, test

    def pre_feature_selection(self, train):
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel

        if self.target_variable == 'return':
            test_features = list(train.columns.drop(['date','ticker', '1_day_change', 'next_day_return', self.target_variable]))
        elif self.target_variable == 'next_day_return':
            test_features = list(train.columns.drop(['date','ticker', self.target_variable]))
        test = train.copy()
        #test['return'] = test['return'].shift(-1)
        #test = test.dropna()
        
        X = test[test_features]
        y = test[self.target_variable]
        
        clf = ExtraTreesRegressor(n_estimators=50, random_state=7)
        clf = clf.fit(X, y)
        best_features = pd.DataFrame([X.columns, clf.feature_importances_]).T
        best_features.columns = ['features', 'score']
        best_features = best_features.sort_values(by=['score'])
        self.best_features = list(best_features['features'].tail(40))
    

    def run_pipeline(self, train, test):

        X = train[['return']+self.test_features]
        y = train[self.target_variable]
        """
        self.bins     = np.linspace(train['return'].min(), train['return'].max(), 3)
        y = np.digitize(y, self.bins)

        y_future_test = np.digitize(test[self.target_variable], self.bins)
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 7)
        

        
        
        max_score = -np.inf
        

        
        self.best_pipeline = None

        for pca_n_components in range(2,25):
            for i in range(20):

                shuffle(self.test_features)
                
                this_features = self.test_features[0:self.k_features]

                pipe_pca = make_pipeline(StandardScaler(),
                                PrincipalComponentAnalysis(n_components=pca_n_components),
                                #mix.GaussianMixture (n_components=3, random_state=7),
                                KNeighborsRegressor(n_neighbors=self.k_neighbors, weights='distance'),
                                )
                
                pipe_pca.fit(X_train[ ['return']+this_features ], y_train)
                
                score = pipe_pca.score(X_test[ ['return']+this_features ], y_test)
                
                test['state'] = pipe_pca.predict(test[['return']+this_features])
                test['next_change'] = test['return'].shift(-1)
                correl = test[['state','next_change']].dropna().corr()['state']['next_change']
                
                if score>max_score and correl>0:
                    
                    
                    self.training_score = pipe_pca.score(X_train[ ['return']+this_features ], y_train)*100
                    self.testing_score = pipe_pca.score(X_test[ ['return']+this_features ], y_test)*100
                    
                    self.future_testing_score = pipe_pca.score(test[ ['return']+this_features ],test[self.target_variable])*100
                    #print(self.training_score)
                    self.pca_n_components = pca_n_components
                    self.best_pipeline = pipe_pca
                    self.found_best_features = ['return'] + this_features
                    max_score = score
                    #print(i)
                    #print('Transf. training accyracy: %.2f%%' % (self.training_score))
                    print('Transf. test accyracy: %.2f%%' % (self.testing_score))
                    print('Future test accyracy: %.2f%%' % (self.future_testing_score))
                    input()

def run_class(params):
    print(params)
    model_generator(params)

if __name__ == '__main__':

    

 
    k_features = range(4,17)
    k_neighbors = range(4,15)
    train_length = range(2, 5)
    params_products = list( product( k_features, k_neighbors, train_length ) )
    print('total number of param sets', len(params_products))
    input()
    params_list = []
    for k_features, k_neighbors, train_length in params_products:

        params = {}
        params['k_features'] = k_features
        params['k_neighbors'] = k_neighbors
        params['train_length'] = train_length
        params_list.append(params)

    shuffle(params_list)
    run_class(params_list[2])
    #p = Pool(1)
    #p.map(run_class, params_list)
    
    
