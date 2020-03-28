
import yfinance
from ta_indicators import get_ta
import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from scipy import stats
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
#from backtest_trader import Backtest, MyStrat
from strategy import setup_strategy
import sklearn.mixture as mix
import numpy as np
import seaborn as sns
import random
from random import randint
import warnings
from datetime import timedelta
import os
from sklearn.svm import SVC
import time
from strategy import MyStrategy, MyStrategy_2, MyStrategy_3
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
def run_decision_tree(train, test_cols):
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['next_classification'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances').tail(10)
    #print(df)
    
    starting_features = list(df['feature'].values)
    #top_starting_features = list(df.sort_values(by='importances').tail(8)['feature'].values)
    return starting_features

history = yfinance.Ticker('QQQ').history(period = '10y', auto_adjust=False).reset_index()
history = get_ta(history, volume=True, pattern=False)
history.columns = map(str.lower, history.columns)

history['return'] = history['close'].pct_change()
history['next_return'] = history['return'].shift(-1)
history['next_classification'] = np.where( history['next_return']>0.005, 1, 0)
history = history.dropna()


dict_classifiers = {
    'lr': LogisticRegression(solver='lbfgs', max_iter=5000),
    'nearest': KNeighborsClassifier(),
    'svm': SVC(gamma='auto'),
    'gbc_2': GradientBoostingClassifier(),
    'gbc': xgb.XGBClassifier(),
    'dt': tree.DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=150),
    "Neural Net": MLPClassifier(solver='adam', alpha=0.0001,learning_rate='constant', learning_rate_init=0.001),
    'nb': GaussianNB()

}


test_cols = list(history.columns.drop(['date','return', 'next_return', 'next_classification']))

max_score = 0

for key, classifier in dict_classifiers.items():
    for scaler in [ MinMaxScaler(feature_range = (0, 1)), StandardScaler() ]:
        for with_pca in [True, False]:
            
            scores = []
            for i in range(10):

                features = run_decision_tree(history.sample(252*3), test_cols)
                if with_pca:
                    svc_pipeline = make_pipeline(scaler,
                                        PrincipalComponentAnalysis(n_components=4),
                                        classifier,
                                        )
                else:
                    svc_pipeline = make_pipeline(scaler,
                                        classifier,
                                        )
                #print(features)
                X_train, X_test, y_train, y_test = train_test_split(history[features], history['next_classification'], test_size=0.33)

                

                svc_pipeline.fit(X_train, y_train)
                train_score = svc_pipeline.score(X_train, y_train) 
                test_score = svc_pipeline.score(X_test, y_test)
                scores.append( [train_score, test_score] )
                train_score, test_score = pd.DataFrame(scores, columns = ['train_score', 'test_score']).mean().round(3).values
                print( key,'\t',  train_score, test_score)
            print('\n')
            if test_score>max_score:
                max_score = test_score
                print('\n')
                print('best model')
                print(svc_pipeline)
                print(test_score)
                print('\n')
            