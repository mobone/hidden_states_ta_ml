from utils import get_data, get_industry_tickers, walk_timeline, plot_results
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
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

class model_generator():
    def __init__(self, data, params):
        # TODO: test different time scales
        # TODO: make class return results for multiple years, check for consistency 
        # TODO: strip down to bare minimum
        # TODO: combine trading with other momentum signals
        #best_features = ['ad', 'adj close', 'adosc', 'apo', 'atr', 'bbands_lower', 'bbands_middle', 'beta', 'dx', 'fama', 'high', 'ht_dcphase', 'ht_leadsine', 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_trendmode', 'kama', 'linearreg_angle', 'low', 'ma', 'macd', 'macd_hist', 'macdfix', 'macdfix_hist', 'mfi', 'minus_dm', 'mom', 'natr', 'open', 'plus_dm', 'ppo', 'rocp', 'rocr100', 'stddev', 't3', 'tema', 'trange', 'trix', 'tsf', 'ultosc', 'willr', 'wma']
        self.conn = sqlite3.connect('markov_models.db')
        self.data = data

        #print(params)

        # iterate through these params
        self.name = params.get('name', namegenerator.gen())
        self.scoring = params.get('scoring', 'r2')
        self.k_features = params.get('k_features', 6)
        self.max_depth = params.get('max_depth', 6)
        self.with_original = params.get('with_original', True)
        self.period = params.get('period', 'max')
        #self.without_bad_year = params.get('without_bad_year', False)
        self.without_bad_year = False
        
        self.test_dates = [
                            {'year': 1, 'start_date': '2017-01-01', 'end_date': '2019-12-31'},
                            #{'year': 2, 'start_date': '2018-01-01', 'end_date': '2018-12-31'},
                            #{'year': 3, 'start_date': '2019-01-01', 'end_date': '2019-12-31'},
                          ]

        self.run_generator()

    def run_generator(self):

        results = []
        #all_features = []
        for self.this_test_dates in self.test_dates:
            self.get_train_test()
            # get features
            self.run_decision_tree()
            #if self.with_original:
            #    self.features = list(set(self.features + ['range', 'close']))
            print(self.features)
            #self.features = ['return', 'range', 'close','volume']
            #all_features.append(self.features)

            self.generate_markov_model()
            #self.get_results()
            
            #results.append(self.result_df)
    
        """
        self.result_df = pd.concat(results)
        #if len(self.test['state'].unique())<3 or float(self.result_df['good_var'])>0.00005:
        #    return
        #filename = float(self.result_df['good_mean']) - (2 * float(self.result_df['good_var']))
        #filename = str(round(float(self.result_df['distance']*100),4))+'_'+str(round(float(self.result_df['good_mean']*100),2))
        #filename = stdev([float(self.result_df['good_mean']), float(self.result_df['good_var'])])
        #filename = round(filename, 4)
        plot_results(self.test.reset_index(), str(filename)+'_'+self.name)

        #self.result_df.index = ['year_1', 'year_2', 'year_3']
        #self.result_df.index = ['year_1', 'year_2', 'year_3']
        
        self.result_df['k_features'] = self.k_features
        self.result_df['max_depth'] = self.max_depth
        self.result_df['scoring'] = self.scoring
        self.result_df['with_original'] = self.with_original
        self.result_df['features'] = str(all_features[0])
        #self.result_df['features_2'] = str(all_features[1])
        #self.result_df['features_3'] = str(all_features[2])
        self.result_df['period'] = self.period
        self.result_df['without_bad_year'] = self.without_bad_year
        self.result_df['iterations'] = self.model.n_iter_
        self.result_df['lower_bound'] = self.model.lower_bound_
        

        self.result_df['name'] = self.name
        print(self.result_df)
        self.result_df.to_sql('models_single_year', self.conn, if_exists='append')
        """
        

    def get_train_test(self):
        self.data['date'] = pd.to_datetime(self.data['date'])

        self.train = self.data[self.data['date']<self.this_test_dates['start_date']]
        self.test = self.data[ (self.data['date']>self.this_test_dates['start_date']) & (self.data['date']<self.this_test_dates['end_date']) ]
        
        
        if self.without_bad_year and len(self.train[self.train['date']>'2018-01-01']):
            self.train = self.train[ (self.train['date']<'2018-01-01') | (self.train['date']>'2018-12-31') ]


    def run_decision_tree(self):    
        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        
        sfs = SFS(clf, 
                k_features=self.k_features, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=-1,
                cv=4)
        test_features = self.train.columns
        test_features = list(test_features.drop(['date', 'ticker', 'return']))
        
        #self.train['target'] = self.train['close'].shift(-1) / self.train['close'] - 1
        self.train = self.train.dropna()
        sfs = sfs.fit(self.train[test_features], self.train['return'])
        
        self.score = sfs.k_score_
        
        self.features = list(sfs.k_feature_names_)


    def generate_markov_model(self):
        # use percent return and try to minimize variance and maximize return
                # TODO: generate model every day
        self.model = mix.GaussianMixture (n_components=3, 
                                    covariance_type="full", #tied
                                    random_state=7,
                                    n_init=60)
        if 'date' in self.train.columns:
            self.train = self.train.set_index('date')
        self.model.fit( self.train[ ['return'] + self.features] )
        if 'date' in self.test.columns:
            self.test = self.test.set_index('date')
        # TODO rename state with english text
        self.test['state'] = self.model.predict(self.test[['return'] + self.features])

        # get next day percent change
        self.test['next_day_change'] = self.test['close'].shift(-1) / self.test['close'] - 1
        #self.test['close'] = self.test['close'].shift(-1)
        print(self.test)
        import pdb; pdb.set_trace()
        self.test.to_csv('test.csv')
        # find the best state numbers
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
        
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state').sort_values(by=['mean'])
        
        result_df['state_names'] = ['sell','buy','strong_buy']

        self.result_df = result_df
        
        print(self.result_df)
        for i in self.result_df.index:
            group = self.test[self.test['state']==i]['next_day_change']
            print(i, group.mean(), group.std())
        #for g, group in self.test.groupby(by='state'):
        #    print(g, group['next_day_change'].mean(), group['next_day_change'].std())
        
        states_used = result_df.index
        self.test['close'] = self.test['close'].shift(-1)
        plot_results(self.test.reset_index(), self.name, result_df.index)

        
        """
        
        self.bad_state, self.middle_state, self.good_state = result_df.index.values
        
        # rename states in test data
        for state_num in [self.good_state, self.middle_state, self.bad_state]:
            self.test.loc[self.test['state']==self.good_state,'state'] = 'strong buy'
            self.test.loc[self.test['state']==self.middle_state,'state'] = 'buy'
            self.test.loc[self.test['state']==self.bad_state,'state'] = 'sell'

        this_columns = list(set(['return']+self.features+['open', 'close','state']))
        
        this_df = self.test[ ['open', 'close', 'return', 'state'] ]
        this_df['name'] = self.name
        this_df['year'] = self.this_test_dates['year']
        this_df.to_sql('trades', self.conn, if_exists='append')
        """
        

    def get_results(self):
        result_df = self.result_df
        print(self.result_df)
        # TODO: CLEAN THIS UP
        good_result = result_df.loc[self.good_state]
        middle_result = result_df.loc[self.middle_state]
        bad_result = result_df.loc[self.bad_state]
        
        good_result.index = ['good_mean', 'good_var']
        middle_result.index = ['middle_mean', 'middle_var']
        bad_result.index = ['bad_mean', 'bad_var']

        good_result=good_result.rename('0')
        middle_result=middle_result.rename('0')
        bad_result=bad_result.rename('0')

        result_df = pd.concat([good_result, middle_result, bad_result])
        result_df = pd.DataFrame(result_df).T
        
        result_df['distance'] = result_df['good_mean'] + (result_df['bad_mean'] * -1)
        
        
        self.result_df = result_df
        #print(self.result_df)
        


def generate_models(params_dicts):
    params = params_dicts
    period = params.get('period', 'max')
    # todo: move get_data outside this method
    tickers = ['QQQ']
    data = get_data(tickers,period=period,pattern=False)
    
    start_time = time()
    model_generator(data, params)
    end_time = time()
    print(params, round( end_time - start_time ,4 ))

if __name__ == '__main__':
    

    k_features = range(4,11)
    max_depth = range(3,11)
    #k_features = range(20,25)
    #max_depth = range(5,8)
    k_features = [6]
    max_depth = [6]
    scoring = ['r2', 'neg_mean_squared_error']
    with_original = [True, False]
    period = ['5y','10y','15y', 'max']
    #without_bad_year = [True, False]

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

    
    for param_group in params_dicts:
        generate_models(param_group)
    #    input()    
    