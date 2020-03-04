import yfinance
import pandas as pd
from hmmlearn.hmm import GaussianHMM,GMMHMM
import matplotlib.pyplot as plt
from ta_indicators import get_ta
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.feature_extraction import PrincipalComponentAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from random import shuffle
from itertools import product
import numpy as np
import warnings
import numpy as np
import namegenerator
import matplotlib
import sqlite3
from multiprocessing import Pool, cpu_count
from simple_trader_hmm import trader
from datetime import timedelta
warnings.simplefilter("ignore")

class pipeline():
    def __init__(self, params_dict = None, model_name = None):
        self.conn = sqlite3.connect('hmm_rolling.db')
        print(params_dict)
        if model_name is None:
            self.name = params_dict.get('name', namegenerator.gen())
            self.pca_n_components = params_dict.get('pca_n_components', 3)
            self.k_features = params_dict.get('k_features', 3)
            self.cutoff_date =  params_dict.get('cutoff_date', '2017-01-01')
            self.train_length =  params_dict.get('train_length', 3)
        elif model_name is not None:
            self.name = model_name
            pca_n_components, k_features, features_found = self.get_model_from_db()
            
            self.pca_n_components = pca_n_components
            self.k_features = k_features
            self.features_found = features_found

        
        #self.cutoff_date = '2017-01-01'
        
        self.n_experiments = 30

        self.get_data()

        if model_name is not None:
            # run production
            print('running production')
            self.run_pipeline(production=True)
            print('predicting completed')
        else:
            self.run_decision_tree()
            self.run_pipeline()
            if self.pipeline_failed == False:
                print('found model', self.features_found)
                self.get_results()
                self.store_results()

    def get_model_from_db(self):
        sql = 'select * from models_final where name == "%s"' % self.name
        df = pd.read_sql(sql, self.conn)
        pca_n_components = int(df['pca_n_components'].values[0])
        k_features = int(df['k_features'].values[0])
        features_found = eval(df['features'].values[0])
        return pca_n_components, k_features, features_found


    def store_results(self):
        self.results_df['name'] = self.name
        self.results_df['cutoff_date'] = str(self.cutoff_date)
        self.results_df['train_score'] = self.train_score
        self.results_df['test_score'] = self.test_score
        self.results_df['train_correl'] = self.train_correl
        self.results_df['test_correl'] = self.test_correl
        self.results_df['pca_n_components'] = self.pca_n_components
        self.results_df['k_features'] = self.k_features
        self.results_df['traing_length'] = self.train_length
        self.results_df['features'] = str(self.features_found)
        
        
        self.results_df.to_sql('models_final', self.conn, if_exists='append', index=False)
        print(self.results_df)
        
        #input()
        


    def run_pipeline(self, production=False):
        self.pipeline_failed = True
        self.max_score = -np.inf
        self.max_correl = -np.inf
        
        # create pipeline
        pipe_pca = make_pipeline(StandardScaler(),
                                PrincipalComponentAnalysis(n_components=self.pca_n_components),
                                #GMMHMM(n_components=3, covariance_type='full'))
                                GaussianHMM(n_components=3, covariance_type='full'))
        exp_num = 0
        while exp_num < self.n_experiments:


            train = self.clean_train.copy()
            test = self.clean_test.copy()
            means = []
            stddevs = []
            scores = []
            correls = []

            if production == False:
                # choose features
                shuffle(self.starting_features)
                test_cols = ['return'] + self.starting_features[0:self.k_features]
                
                if 'stoch' not in str(test_cols):
                    continue
                
            elif production == True:
                test_cols = self.features_found

            
            # test features on training dataset
            pipe_pca.fit(train[ test_cols ])
            try:
                self.train_score = np.exp( pipe_pca.score(train[ test_cols ]) / len(train) ) * 100
            except:
                self.train_score = None
            train['state'] = pipe_pca.predict(train[test_cols])
            train = self.rename_states(train)
            if train is None:
                continue
            criteria_check = self.check_criteria(train)
            if criteria_check == False:
                continue

            # get the correlation between state and next day percent changes
            train['next_day'] = train['close'].shift(-1) / train['close'] - 1
            train_means = train.dropna().groupby(by='state')[['return', 'next_day']].mean()
            train_correl = train_means.corr()
            self.train_correl = train_correl['return']['next_day']


            # do the same for the test data
            pipe_pca.fit(test[ test_cols ])
            try:
                self.test_score = np.exp( pipe_pca.score(test[ test_cols ]) / len(test) ) * 100
            except:
                self.test_score = None
            test['state'] = pipe_pca.predict(test[test_cols])
            test = self.rename_states(test)

            if production == True:
                self.new_predictions = test.tail(30)
                return

            if test is None:
                continue
            criteria_check = self.check_criteria(test)
            if criteria_check == False:
                continue

            # get the correlation between state and next day percent changes
            test['next_day'] = test['close'].shift(-1) / test['close'] - 1
            test_means = test.dropna().groupby(by='state')[['return', 'next_day']].mean()
            test_correl = test_means.corr()
            self.test_correl = test_correl['return']['next_day']

            exp_num = exp_num + 1
            
            if self.train_correl > self.max_correl and self.test_correl>0:
                
                self.train_predicted = train
                self.test_predicted = test
                self.features_found = test_cols

                self.train_means = train_means
                self.test_means = test_means
                

                #print('model found on expirement number', exp_num)
                #print(self.features_found)
                
                self.max_correl = self.train_correl
                self.pipeline_failed = False
                
                

    def check_criteria(self, df):

        counts_check = int( len(df) *.1 )
        # ensure all three states are used 
        counts = df.groupby(by='state')['state'].count()
        if not counts[counts<counts_check].empty or len(df.groupby(by='state'))<3:
            return False

        # get the correlation between state and next day percent changes
        df['next_day'] = df['close'].shift(-1) / df['close'] - 1
        correl = df.dropna().groupby(by='state')[['return', 'next_day']].mean().corr()
        
        correl = correl['return']['next_day']

        if correl<.5:
            return False

        # ensure buy and strong buy returns and next returns are positive
        means = df.dropna().groupby(by='state')[['return', 'next_day']].mean()
        
        if float(means[means.index == 0]['return'])>0 or float(means[means.index == 1]['return'])<0 or float(means[means.index == 2]['return'])<0:
            return False
        if float(means[means.index == 0]['return'])>0 or float(means[means.index == 1]['next_day'])<0 or float(means[means.index == 2]['next_day'])<0:
            return False
        
        return True

    def rename_states(self, df):
        # rename states accordingly
        state_nums = df.groupby(by='state').mean().sort_values(by='return')
        if len(state_nums)<3:
            return None

        state_nums['renamed_state'] = ['sell', 'buy', 'strong_buy']
        state_nums['renamed_state_nums'] = [0,1,2]
        
        for i in list(state_nums.index):
            #print('renaming state %s to state %s' % ( i, state_nums.loc[i]['renamed_state']))
            df.loc[df['state']==i, 'renamed_state'] = state_nums.loc[i]['renamed_state']
            df.loc[df['state']==i, 'renamed_state_num'] = state_nums.loc[i]['renamed_state_nums']

        df['state'] = df['renamed_state_num']
        #del df['renamed_state_num']

        return df

    def get_results(self):
        df = pd.DataFrame()
        
        self.train_means = self.train_means.unstack().to_frame().T
        self.train_means.columns = ['train_0_return', 'train_1_return', 'train_2_return', 'train_0_next_day', 'train_1_next_day', 'train_2_next_day', ]
        self.test_means = self.test_means.unstack().to_frame().T
        self.test_means.columns = ['test_0_return', 'test_1_return', 'test_2_return', 'test_0_next_day', 'test_1_next_day', 'test_2_next_day', ]

        self.results_df = pd.concat([self.train_means, self.test_means],axis=1)

        # store trades to be used in trader
        self.test_predicted['name'] = self.name
        self.test_predicted.to_sql('trades_final', self.conn, if_exists='append', index=False)
        
        safe_return = trader(self.name, 'QQQ', 'QLD').return_percentage
        moderate_return = trader(self.name, 'QQQ', 'TQQQ').return_percentage
        extreme_return = trader(self.name, 'QLD', 'TQQQ').return_percentage

        self.results_df['safe_return'] = safe_return
        self.results_df['moderate_return'] = moderate_return
        self.results_df['extreme_return'] = extreme_return

        self.test_predicted.plot.scatter(x='date',
                                        y='close',
                                        c='renamed_state_num',
                                        colormap='viridis')

                    
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        file_name = self.name+'_'+str(self.cutoff_date).replace('-','')
        plt.savefig('./plots_rolling/%s.png'% file_name )


    def get_data(self):
        stock_df = yfinance.Ticker('QQQ').history(period='max')
        stock_df = stock_df.reset_index()
        stock_df = get_ta(stock_df, volume=True, pattern=False)
        stock_df.columns = map(str.lower, stock_df.columns)
        stock_df["return"] = stock_df["close"].pct_change()
        stock_df.columns = map(str.lower, stock_df.columns)
        stock_df = stock_df.dropna()

        cutoff_datetime = pd.to_datetime(self.cutoff_date)

        start_train = cutoff_datetime - timedelta(days=365 * self.train_length)
        end_train = cutoff_datetime

        start_test = cutoff_datetime
        end_test = cutoff_datetime + timedelta(days=365 * 1)

        self.clean_train = stock_df[ (stock_df['date'] > start_train) & (stock_df['date'] < end_train) ]
        self.clean_test = stock_df[ (stock_df['date'] > start_test) & (stock_df['date'] < end_test) ]
        

    def run_decision_tree(self):
        
        # get features
        train = self.clean_train.copy()
        test = self.clean_test.copy()
        test_cols = train.columns.drop(['date','return'])
        X = train[test_cols]
        y = train['return']
        clf = ExtraTreesRegressor(n_estimators=50)
        clf = clf.fit(X, y)
        df = pd.DataFrame([test_cols, clf.feature_importances_]).T
        df.columns = ['feature', 'importances']
        df = df.sort_values(by='importances').tail(40)
        
        self.starting_features = list(df['feature'].tail(40).values)
        

    

def run_pipeline_class(params_dict):
    
    pipeline(params_dict = params_dict)



if __name__ == '__main__':
    # create params list to iterate through
    #pca_n_components = list( range(3,15) )
    params_list_of_dicts = []
    pca_n_components = [3]
    simulations = [0,1,2,3]
    k_features = list( range(3,15) )

    #pca_n_components = [3]
    #simulations = [0,]
    #k_features = list( range(3,4) )

    cutoff_dates = ['2017-01-01', '2018-01-01', '2019-01-01', ] 
    train_length = [2,3,4,5]

    params_list = list(product( pca_n_components, k_features, train_length, simulations ))
    

    shuffle(params_list)
    
    for pca_n_components, k_features, train_length, _ in params_list:
        name = namegenerator.gen()
        for cutoff_date in cutoff_dates:
            params_dict = { 
                            'pca_n_components': pca_n_components, 
                            'k_features': k_features,
                            'cutoff_date': cutoff_date, 
                            'train_length': train_length,
                            'name': name}
            params_list_of_dicts.append(params_dict)

    p = Pool( int(cpu_count()-1) )
    p.map(run_pipeline_class, params_list_of_dicts)
    #run_pipeline_class(params_list_of_dicts[0])
    