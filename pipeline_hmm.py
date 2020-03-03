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
from multiprocessing import Pool
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
        try:
            self.store_results()
        except Exception as e:
            print(e)

    def store_results(self):
        output_dict = {}
        output_dict['name'] = self.name
        output_dict['probability'] = self.probability
        output_dict['correl'] = self.correl
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
        self.predicted_test['name'] = self.name
        self.predicted_test.to_sql('trades_with_test', self.conn, if_exists='append', index=False)
        #input()
        


    def run_pipeline(self):
        
        max_score = -np.inf
        max_correl = -np.inf
        
                                                                                        #for pca_n_components, k_features in params_list:
        for simulations in range(self.n_experiments):


            train = self.clean_train.copy()
            test = self.clean_test.copy()

            X_train, X_test, y_train, y_test = train_test_split(train, train['return'], random_state=7, test_size=0.3)

            # choose features
            shuffle(self.starting_features)
            test_cols = ['return'] + self.starting_features[0:self.k_features]

            # create pipeline
            pipe_pca = make_pipeline(StandardScaler(),
                                    PrincipalComponentAnalysis(n_components=self.pca_n_components),
                                    #GMMHMM(n_components=3, covariance_type='full'))
                                    GaussianHMM(n_components=3, covariance_type='full'))


            # train model, get probability score, and predict new states
            """
            try:
                pipe_pca.fit(train[ test_cols ])
                score = np.exp( pipe_pca.score(test[test_cols])/len(test) )*100
                test['state'] = pipe_pca.predict(test[test_cols ])
            except Exception as e:
                print(e)
                continue
            """

            
            
            try:
                pipe_pca.fit(X_train[ test_cols ])
                score = np.exp( pipe_pca.score(X_test[ test_cols ]) / len(X_test) ) * 100
                #score = np.exp( pipe_pca.score(test[test_cols])/len(test) )*100
                X_test['state'] = pipe_pca.predict(X_test[ test_cols ])
            except Exception as e:
                print(e)
                continue

            # parse results
            """
            counts = test.groupby(by='state')['state'].count()
            if not counts[counts<50].empty:
                continue
            if len(test.groupby(by='state'))<3:
                continue
            
            test['next_day'] = test['close'].shift(-1) / test['close'] - 1

            correl = test.dropna().groupby(by='state')[['return', 'next_day']].mean().corr()
            
            correl = correl['return']['next_day']            
            """
            counts = X_test.groupby(by='state')['state'].count()
            if not counts[counts<50].empty:
                continue
            if len(X_test.groupby(by='state'))<3:
                continue
            
            X_test['next_day'] = X_test['close'].shift(-1) / X_test['close'] - 1

            correl = X_test.dropna().groupby(by='state')[['return', 'next_day']].mean().corr()
            
            correl = correl['return']['next_day']
            
            
            if score > max_score and correl > max_correl:
                


                test['state'] = pipe_pca.predict(test[test_cols])
                counts = test.groupby(by='state')['state'].count()
                if not counts[counts<50].empty:
                    continue
                if len(test.groupby(by='state'))<3:
                    continue

                max_score = score
                max_correl = correl
                
                self.probability = score
                self.correl = correl

                test['next_day'] = test['close'].shift(-1) / test['close'] - 1

                # rename states accordingly
                state_nums = test.groupby(by='state').mean().sort_values(by='return')
                state_nums['renamed_state'] = [0,1,2]
                #print(state_nums[ ['return', 'next_day', 'renamed_state'] ])
                for i in list(state_nums.index):
                    #print('renaming state %s to state %s' % ( i, int(state_nums.loc[i]['renamed_state'])))
                    test.loc[test['state']==i, 'renamed_state'] = int(state_nums.loc[i]['renamed_state'])

                test['state'] = test['renamed_state']
                del test['renamed_state']
                """
                print()
                print('features', test_cols)
                print('score', score)
                print('correl', correl)
                print()
                """
                
                self.means = test.groupby(by='state')['return'].mean()
                if len(self.means.values)<3:
                    continue
                self.state_0_mean, self.state_1_mean, self.state_2_mean = self.means.values
                self.means = test.dropna().groupby(by='state')['next_day'].mean()
                self.state_0_next_mean, self.state_1_next_mean, self.state_2_next_mean = self.means.values
                
                

                self.predicted_test = test
                self.features_found = test_cols
                self.predicted_test.plot.scatter(x='date',
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

    p = Pool(15)
    p.map(run_pipeline_class, params_list)
    #pca_n_components, k_features = params_list[0]
    #pipeline(pca_n_components, k_features)
