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
from tiny_pipeline import pipeline
conn = sqlite3.connect('tiny_pipeline.db')

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

if __name__ == '__main__':
    symbol = 'AAPL'
    train, test, test_cols = get_data(symbol)
    sql = 'select features, name from models group by name order by "safe_return" desc limit 10'
    features_list = pd.read_sql(sql, conn)
    
    for index, row in features_list.iterrows():
        print(row)
        features = eval(row[0])
        model_name = row[1]
        print(features, model_name)
        
        pipeline(train, test, features, model_name=model_name)

    