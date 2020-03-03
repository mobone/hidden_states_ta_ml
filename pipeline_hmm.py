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
warnings.simplefilter("ignore")

class pipeline():
    def __init__(self, pca_n_components, k_features):
        self.conn = sqlite3.connect('hmm.db')
        self.name = namegenerator.gen()
        
        self.cutoff_date = '2017-01-01'
        self.pca_n_components = pca_n_components
        self.k_features = k_features
        self.n_experiments = 35

        self.get_data()
        self.run_decision_tree()
        self.run_pipeline()
        if self.failed_real_test == False:
            self.get_results()
            self.store_results()


    def store_results(self):
        output_dict = {}
        output_dict['name'] = self.name
        output_dict['expirement_score'] = self.max_score
        output_dict['expirement_correl'] = self.max_correl
        output_dict['real_test_score'] = self.real_test_score
        output_dict['real_test_correl'] = self.real_test_correl
        output_dict['state_0_mean'] = self.state_0_mean * 100
        output_dict['state_1_mean'] = self.state_1_mean * 100
        output_dict['state_2_mean'] = self.state_2_mean * 100
        output_dict['state_0_next_mean'] = self.state_0_next_mean * 100
        output_dict['state_1_next_mean'] = self.state_1_next_mean * 100
        output_dict['state_2_next_mean'] = self.state_2_next_mean * 100

        output_dict['pca_n_components'] = self.pca_n_components
        output_dict['k_features'] = self.k_features
        output_dict['features'] = str(self.features_found)

        df = pd.DataFrame.from_dict(output_dict, orient='index').T
        df.to_sql('models_with_test', self.conn, if_exists='append', index=False)
        print(df)
        self.test_predicted['name'] = self.name
        self.test_predicted.to_sql('trades_with_test', self.conn, if_exists='append', index=False)
        #input()
        


    def run_pipeline(self):
        
        self.max_score = -np.inf
        self.max_correl = -np.inf
        
        for exp_num in range(self.n_experiments):


            train = self.clean_train.copy()
            test = self.clean_test.copy()
            scores = []
            correls = []

            # choose features
            shuffle(self.starting_features)
            test_cols = ['return'] + self.starting_features[0:self.k_features]

            for cv_split_num in range(3):
                X_train, X_test, y_train, y_test = train_test_split(train, train['return'], test_size=0.3)


                # create pipeline
                pipe_pca = make_pipeline(StandardScaler(),
                                        PrincipalComponentAnalysis(n_components=self.pca_n_components),
                                        #GMMHMM(n_components=3, covariance_type='full'))
                                        GaussianHMM(n_components=3, covariance_type='full'))


                # train model, get probability score, and predict new states    
                try:
                    pipe_pca.fit(X_train[ test_cols ])

                    # get the probability score
                    score = np.exp( pipe_pca.score(X_test[ test_cols ]) / len(X_test) ) * 100
                    
                    # predict the new states
                    X_test['state'] = pipe_pca.predict(X_test[ test_cols ])

                    # ensure all three states are used 
                    counts = X_test.groupby(by='state')['state'].count()
                    if not counts[counts<10].empty or len(X_test.groupby(by='state'))<3:
                        continue

                    # get the correlation between state and next day percent changes
                    X_test['next_day'] = X_test['close'].shift(-1) / X_test['close'] - 1
                    correl = X_test.dropna().groupby(by='state')[['return', 'next_day']].mean().corr()
                    correl = correl['return']['next_day']

                    scores.append(score)
                    correls.append(correl)
                except Exception as e:
                    print(e)
                    continue

            if len(scores)<3:
                continue
            
            score = np.mean(scores)
            correl = np.mean(correls)            
            if score > self.max_score and correl > self.max_correl and correl>0:
                
                


                test['state'] = pipe_pca.predict(test[test_cols])
                
                counts = test.groupby(by='state')['state'].count()
                if not counts[counts<50].empty or len(test.groupby(by='state'))<3:
                    self.failed_real_test = True
                    #print('failed real world test')
                    continue

                # get the correlation between state and next day percent changes
                test['next_day'] = test['close'].shift(-1) / test['close'] - 1
                real_test_correl = test.dropna().groupby(by='state')[['return', 'next_day']].mean().corr()
                real_test_correl = real_test_correl['return']['next_day']
                if real_test_correl<0:
                    #print('failed real world test')
                    continue
                self.real_test_correl = real_test_correl

                # get the score for the real test data
                self.real_test_score = np.exp( pipe_pca.score(test[ test_cols ]) / len(test) ) * 100

                print('experiment %s found model' % exp_num, score, correl, test_cols)
                self.test_predicted = test
                self.features_found = test_cols
                self.max_score = score
                self.max_correl = correl
                
                self.probability = score
                self.correl = correl
                self.failed_real_test = False

    def get_results(self):
        
        self.test_predicted['next_day'] = self.test_predicted['close'].shift(-1) / self.test_predicted['close'] - 1

        # rename states accordingly
        state_nums = self.test_predicted.groupby(by='state').mean().sort_values(by='return')
        state_nums['renamed_state'] = ['sell', 'buy', 'strong_buy']
        state_nums['renamed_state_nums'] = [0,1,2]
        
        for i in list(state_nums.index):
            #print('renaming state %s to state %s' % ( i, state_nums.loc[i]['renamed_state']))
            self.test_predicted.loc[self.test_predicted['state']==i, 'renamed_state'] = state_nums.loc[i]['renamed_state']
            self.test_predicted.loc[self.test_predicted['state']==i, 'renamed_state_num'] = state_nums.loc[i]['renamed_state_nums']

        self.test_predicted['state'] = self.test_predicted['renamed_state_num']
        del self.test_predicted['renamed_state_num']

        self.means = self.test_predicted.groupby(by='renamed_state')['return'].mean()
        self.state_0_mean = self.means['sell']
        self.state_1_mean = self.means['buy']
        self.state_2_mean = self.means['strong_buy']
        
        self.means = self.test_predicted.dropna().groupby(by='renamed_state')['next_day'].mean()
        self.state_0_next_mean = self.means['sell']
        self.state_1_next_mean = self.means['buy']
        self.state_2_next_mean = self.means['strong_buy']
        
        self.test_predicted.plot.scatter(x='date',
                                        y='close',
                                        c='state',
                                        colormap='viridis')
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.savefig('./plots/%s.png'% self.name )


    def get_data(self):
        spy_data = yfinance.Ticker('SPY').history(period='7y')
        spy_data = spy_data.reset_index()
        spy_data = get_ta(spy_data, volume=True, pattern=False)
        spy_data.columns = map(str.lower, spy_data.columns)
        spy_data["return"] = spy_data["close"].pct_change()
        spy_data.columns = map(str.lower, spy_data.columns)
        spy_data = spy_data.dropna()
        self.clean_train = spy_data[spy_data['date']<self.cutoff_date]
        self.clean_test = spy_data[spy_data['date']>self.cutoff_date]

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
        

    

def run_pipeline_class(params_list):
    pca_n_components, k_features = params_list    
    pipeline(pca_n_components, k_features)



if __name__ == '__main__':
    # create params list to iterate through
    pca_n_components = list( range(3,15) )
    k_features = list( range(3,15) )
    params_list = list(product( pca_n_components, k_features ))
    shuffle(params_list)

    p = Pool( int(cpu_count()-1) )
    p.map(run_pipeline_class, params_list)
    #pca_n_components, k_features = params_list[0]
    #pipeline(pca_n_components, k_features)
