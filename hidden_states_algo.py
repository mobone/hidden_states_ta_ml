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

class model_generator():
    def __init__(self, data, params):
        # TODO: test different time scales
        # TODO: make class return results for multiple years, check for consistency 
        # TODO: strip down to bare minimum
        # TODO: combine trading with other momentum signals
        best_features = ['ad', 'adj close', 'adosc', 'apo', 'atr', 'bbands_lower', 'bbands_middle', 'beta', 'dx', 'fama', 'high', 'ht_dcphase', 'ht_leadsine', 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_trendmode', 'kama', 'linearreg_angle', 'low', 'ma', 'macd', 'macd_hist', 'macdfix', 'macdfix_hist', 'mfi', 'minus_dm', 'mom', 'natr', 'open', 'plus_dm', 'ppo', 'rocp', 'rocr100', 'stddev', 't3', 'tema', 'trange', 'trix', 'tsf', 'ultosc', 'willr', 'wma']
        conn = sqlite3.connect('markov_models.db')
        self.data = data
        # iterate through these params

        self.name = params.get('name', namegenerator.gen())
        self.scoring = params.get('scoring', 'r2')
        self.k_features = params.get('k_features', 6)
        self.max_depth = params.get('max_depth', 6)
        self.with_original = params.get('with_original', True)
        self.buy_state = params.get('buy_state', 'good')
        self.period = params.get('period', 'max')

        self.test_dates = ('2017-01-01','2020-02-28')

        # maybe play with these params
        #self.covariance_type = ['diag','full','tied']
        #self.n_components = 3

        self.get_train_test()
        
        self.run_decision_tree()
        
          
        if self.with_original:
            self.features = list(set(self.features + ['range', 'close']))

        for feature in self.features:
            if feature in best_features:
                break
            #print('no good features found')
            return
        
        self.generate_markov_model()
        self.get_results()
        if self.result_df['bad_mean'].values[0]>-0.003:
            return
        elif self.result_df['good_mean'].values[0]<0.008:
            return

        percent_return, benchmark_return = walk_timeline(self.test, self.buy_state, self.good_state, self.middle_state, self.bad_state)
        

        self.result_df['percent_return'] = percent_return
        self.result_df['benchmark_percent_return'] = benchmark_return
        self.result_df['k_features'] = self.k_features
        self.result_df['max_depth'] = self.max_depth
        self.result_df['scoring'] = self.scoring
        self.result_df['with_original'] = self.with_original
        self.result_df['buy_state'] = self.buy_state
        self.result_df['features'] = str(self.features)
        self.result_df['period'] = self.period
        
        self.result_df['name'] = self.name
        print(self.result_df)
        self.result_df.to_sql('models_with_best_features_and_best_means', conn, if_exists='append', index=False)
        if percent_return>.5:
            plot_results(self.model, self.test, self.name)

    def get_train_test(self):
        self.data['date'] = pd.to_datetime(self.data['date'])

        self.train = self.data[self.data['date']<self.test_dates[0]]
        self.test = self.data[ (self.data['date']>self.test_dates[0]) & (self.data['date']<self.test_dates[1]) ]
        #print(self.train)
        #print(self.test)


    def run_decision_tree(self):    
        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        
        sfs = SFS(clf, 
                k_features=self.k_features, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=-1,
                cv=4)
        test_features = self.train.columns.copy()
        test_features = list(test_features.drop(['date', 'ticker', 'return']))

        sfs = sfs.fit(self.train[test_features], self.train['return'])
        
        self.score = sfs.k_score_
        
        self.features = list(sfs.k_feature_names_)
        print('features', self.features)
        

    def run_grid_search(self):
        # find best params and features, using return as the target
        clf = DecisionTreeRegressor(random_state=7, max_depth=4)
        sfs = SFS(clf, 
                k_features=4, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=1,
                cv=4)

        pipe = Pipeline([('sfs', sfs), 
                        ('clf', clf)])
        
        param_grid = [{
           'sfs__k_features': list(range(2,5)),
            'sfs__estimator__max_depth': list(range(3,6))
        }]

        gs = GridSearchCV(estimator=pipe, 
                        param_grid=param_grid, 
                        scoring=self.scoring, 
                        n_jobs=15, 
                        cv=4,
                        verbose=1,
                        iid=True,
                        refit=True)

        test_features = list(self.train.columns.drop(['date', 'ticker', 'return']))

        gs = gs.fit(self.train[test_features], self.train['return'])

        self.score = gs.best_score_
        
        self.features = list(gs.best_estimator_.steps[0][1].k_feature_names_)

        print('params', gs.best_params_)
        print('features', self.features)
        print('score', self.score)

        self.k_features = gs.best_params_['sfs__k_features']
        self.max_depth = gs.best_params_['sfs__estimator__max_depth']
    
    def generate_markov_model(self):
        # use percent return and try to minimize variance and maximize return
                # TODO: generate model every day
        self.model = mix.GaussianMixture(n_components=3, 
                                    covariance_type="full", #tied
                                    random_state=7,
                                    n_init=100)
        self.train = self.train.set_index('date')
        self.model.fit( self.train[ ['return'] + self.features] )

        self.test = self.test.set_index('date')
        # TODO rename state with english text
        self.test['state'] = self.model.predict(self.test[['return'] + self.features])

    def get_results(self):
        #print("Means and vars of each hidden state")
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
            #results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0][0] ])
            
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state')
        
        result_df = result_df.sort_values(by=['mean'])
        # todo: figure out what to do with intermediate state
        self.good_state = int(result_df.tail(1).index.values[0])
        self.middle_state = int(result_df.tail(2).head(1).index.values[0])
        self.bad_state = int(result_df.head(1).index.values[0])

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
        result_df['good_state'] = self.good_state
        result_df['middle_state'] = self.middle_state
        result_df['bad_state'] = self.bad_state
        result_df['distance'] = result_df['good_mean'] + (result_df['bad_mean'] * -1)
        self.result_df = result_df


def generate_models(params_dicts):
    params = params_dicts
    period = params.get('period', 'max')
    # todo: move get_data outside this method
    tickers = ['QQQ']
    data = get_data(tickers,period=period,pattern=False)
    print(params)
    
    model_generator(data, params)

if __name__ == '__main__':
    

    k_features = range(4,15)
    max_depth = range(3,15)
    scoring = ['r2', 'neg_mean_squared_error']
    buy_state = ['good', 'middle']
    with_original = [True, False]
    period = ['5y','10y','15y', 'max']

    params_list = list( product( k_features, max_depth, scoring, buy_state, with_original, period ) )
    print('total number of models', len(params_list))
    shuffle(params_list)
    params_dicts = []
    for k_features, max_depth, scoring, buy_state, with_original, period in params_list:
        params = {}
        params['k_features'] = k_features
        params['max_depth'] = max_depth
        params['scoring'] = scoring
        params['buy_state'] = buy_state
        params['with_original'] = with_original
        params['period'] = period
        params_dicts.append( params )
    
    p = Pool(16)
    p.map(generate_models, params_dicts)

    """
    for param_group in params_dicts:
        generate_models(params_dicts[0])
        input()
    """
    