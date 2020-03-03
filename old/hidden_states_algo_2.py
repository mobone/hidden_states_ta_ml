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


        # TODO: make class return results for multiple years, check for consistency 
        # TODO: strip down to bare minimum
        # TODO: combine trading with other momentum signals
        # TODO: use timedelta to define train dataset, and advance through the years, or # TODO: test different time scales
class model_generator():
    def __init__(self, params):
        self.conn = sqlite3.connect('markov_models.db')
        

        #print(params)

        # iterate through these params
        self.name = params.get('name', namegenerator.gen())
        self.scoring = params.get('scoring', 'max_error')
        self.k_features = params.get('k_features', 25)
        self.max_depth = params.get('max_depth', 2)
        
        self.period = params.get('period', '3y')

        tickers = ['SPY']
        self.data = get_data(tickers,period=self.period)
        
        """
        self.test_dates = [
                            {'year': 1, 'start_date': '2017-01-01', 'end_date': '2017-12-31'},
                            {'year': 2, 'start_date': '2018-01-01', 'end_date': '2018-12-31'},
                            {'year': 3, 'start_date': '2019-01-01', 'end_date': '2019-12-31'},
                          ]
        """
        self.run_generator()

    def run_generator(self):
        # get features to use
        self.this_test_dates = {'year': 1, 'start_date': '2019-01-01', 'end_date': '2020-03-01'}
        self.get_train_test()
        #self.run_grid_search()
        self.get_pca()
        #self.features = self.features + ['mom']
        #print(self.features)

        #self.generate_markov_model_with_pca()
        #self.test_model_with_pca()
        #self.get_results_with_pca()
        self.get_results_with_pca_mew()


    def get_train_test(self):
        self.data['date'] = pd.to_datetime(self.data['date'])

        self.train = self.data[self.data['date']<self.this_test_dates['start_date']]
        self.test = self.data[ (self.data['date']>self.this_test_dates['start_date']) & (self.data['date']<self.this_test_dates['end_date']) ]


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

        self.train = self.train.dropna()
        sfs = sfs.fit(self.train[test_features], self.train['return'])
        
        self.score = sfs.k_score_
        
        self.features = list(sfs.k_feature_names_)

    def run_grid_search(self):
        # find best params and features, using return as the target
        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        #clf = RandomForestRegressor(random_state=7, max_depth=self.max_depth,n_estimators=10)
        sfs = SFS(clf, 
                k_features=4, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=1,
                cv=3)

        pipe = Pipeline([('sfs', sfs), 
                        ('clf', clf)])
        
        param_grid = [{
           'sfs__k_features': list(range(5,9)),
           #'sfs__estimator__max_depth': list(range(2,6)),
           #'sfs__scoring': ['r2','neg_mean_squared_error','max_error'],
        }]

        gs = GridSearchCV(estimator=pipe, 
                        param_grid=param_grid, 
                        scoring=self.scoring, 
                        n_jobs=15, 
                        cv=3,
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

    def get_pca(self):
        from sklearn.decomposition import PCA
        n_components = 3
        
        test_features = list(self.train.columns.drop(['date', 'ticker']))
        X = self.train[test_features]
        y = self.test[test_features]
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_pca = pca.transform(X)
        y_pca = pca.transform(y)
        print("original shape:   ", X.shape)
        print("transformed shape:", X_pca.shape)
        
        print(pca.components_)
        print(pca.explained_variance_)

        feature_names = []
        for i in range(n_components):
            feature_names.append('vec_%s' % i)
        print(feature_names)


        self.X_pca = X_pca
        self.y_pca = y_pca
        self.X_pca = pd.DataFrame(self.X_pca, columns = feature_names)
        self.y_pca = pd.DataFrame(self.y_pca, columns = feature_names)

        self.X_pca['return'] = self.train['return']
        self.X_pca['close'] = self.train['close']
        self.X_pca = self.X_pca[['return'] + feature_names + ['close']]

        self.y_pca['return'] = self.test['return'].values
        self.y_pca['close'] = self.test['close'].values
        self.y_pca['date'] = self.test['date'].values
        self.y_pca = self.y_pca.set_index('date')
        self.y_pca = self.y_pca[['return'] + feature_names + ['close']]
        print(self.X_pca)
        print(self.y_pca)
        

    def get_pca_older(self):
        from sklearn.decomposition import PCA

        self.train['next_change'] = self.train['close'].shift(-1) / self.train['close'] - 1
        self.train = self.train.dropna()
        self.test['next_change'] = self.test['close'].shift(-1) / self.test['close'] - 1
        self.test = self.test.dropna()

        test_features = list(self.train.columns.drop(['date', 'ticker']))
        plt.figure(2, figsize=(8, 6))
        plt.clf()
        X = self.train[test_features].to_numpy()
        
        y = self.train['next_change']
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5


        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                    edgecolor='k')
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        #plt.show()

        # To getter a better understanding of interaction of the dimensions
        # plot the first three PCA dimensions
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(self.train[test_features])

        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
                cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])

        plt.show()

    def get_pca_old(self):
        
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        test_features = list(self.train.columns.drop(['date', 'ticker']))
        pca = decomposition.PCA(n_components=3)
        pca_result = pca.fit(self.train[ test_features ])
        
        self.test['next_change'] = self.test['close'].shift(-1) / self.test['close'] - 1
        self.test = self.test.dropna()

        X = pca.transform( self.test[test_features])
        
        y = pd.qcut( (self.test['next_change']*100).astype(int),
                            q=[0, .2, .4, .6, .8, 1],
                            labels=False,
                            duplicates="drop",
                            precision=2)


        print(y)
        
        for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
            ax.text3D(X[y == label, 0].mean(),
                    X[y == label, 1].mean() + 1.5,
                    X[y == label, 2].mean(), name,
                    horizontalalignment='center',
                    bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
                    edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        plt.show()
        input()

    def generate_markov_model_with_pca(self):
        
        self.model = mix.GaussianMixture (n_components=3, 
                                    covariance_type="full", #tied
                                    random_state=7,
                                    n_init=60)
        #if 'date' in self.train.columns:
        #    self.train = self.train.set_index('date')
        self.model.fit( self.X_pca )
        """
        lowest_bic = -np.infty
        bic = []
        n_components_range = range(2, 7)
        cv_types = ['tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                for weight_concentration_prior_type in ['dirichlet_process','dirichlet_distribution']:
                    # Fit a Gaussian mixture with EM
                    gmm = mix.BayesianGaussianMixture(n_components=n_components,
                                                covariance_type=cv_type, random_state=7, weight_concentration_prior_type = weight_concentration_prior_type)
                    #X = self.train[ ['return'] + self.features]
                    X = self.X_pca
                    gmm.fit(X)
                    bic.append(gmm.lower_bound_)
                    if bic[-1] > lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm
                        print(cv_type, n_components, weight_concentration_prior_type, gmm.lower_bound_)
        
        self.model = best_gmm
        """
        

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
        
        """
        lowest_bic = -np.infty
        bic = []
        n_components_range = range(2, 4)
        cv_types = ['tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                for weight_concentration_prior_type in ['dirichlet_process','dirichlet_distribution']:
                    # Fit a Gaussian mixture with EM
                    gmm = mix.BayesianGaussianMixture(n_components=n_components,
                                                covariance_type=cv_type, random_state=7, weight_concentration_prior_type = weight_concentration_prior_type)
                    X = self.train[ ['return'] + self.features]
                    gmm.fit(X)
                    bic.append(gmm.lower_bound_)
                    if bic[-1] > lowest_bic:
                        lowest_bic = bic[-1]
                        best_gmm = gmm
                        print(cv_type, n_components, weight_concentration_prior_type, gmm.lower_bound_)
        
        self.model = best_gmm
        """
        

    def test_model_with_pca(self):
        self.test = self.y_pca
        self.test['state'] = self.model.predict(self.test)
        #print(self.model.score( self.test ) )
        #print(self.model.score_samples( self.test[['return']+self.features] ))    
        
    def test_model(self):
        
        self.test['state'] = self.model.predict(self.test[['return'] + self.features])
        print(self.model.score( self.test[['return']+self.features]) )
        #print(self.model.score_samples( self.test[['return']+self.features] ))

    
    def get_results_with_pca_new(self):
        """
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
        
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state').sort_values(by=['mean'])
        print(result_df)
        self.test['next_change'] = self.test['close'].shift(-1) / self.test['close'] - 1
        #self.test[ ['date', 'state', 'close', 'next_change'] ].to_csv('test.csv')
        for state in result_df.index:
            this_group = self.test.loc[self.test['state']==state, 'next_change']
            print(state, float(this_group.mean()), float(this_group.std()))
        """
        self.test = self.y_pca
        plot_results(self.test, self.name, result_df.index)
    
    def get_results_with_pca(self):
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
        
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state').sort_values(by=['mean'])
        print(result_df)
        self.test['next_change'] = self.test['close'].shift(-1) / self.test['close'] - 1
        #self.test[ ['date', 'state', 'close', 'next_change'] ].to_csv('test.csv')
        for state in result_df.index:
            this_group = self.test.loc[self.test['state']==state, 'next_change']
            print(state, float(this_group.mean()), float(this_group.std()))

        plot_results(self.test, self.name, result_df.index)


    def get_results(self):
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
        
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state').sort_values(by=['mean'])
        print(result_df)
        self.test['next_change'] = self.test['close'].shift(-1) / self.test['close'] - 1
        #self.test[ ['date', 'state', 'close', 'next_change'] ].to_csv('test.csv')
        for state in result_df.index:
            this_group = self.test.loc[self.test['state']==state, 'next_change']
            print(state, float(this_group.mean()), float(this_group.std()))

        plot_results(self.test, self.name, result_df.index)

        """
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
    """ 


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
    