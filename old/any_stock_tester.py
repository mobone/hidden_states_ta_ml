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
from tiny_pipeline_stripped import pipeline
import requests
import re
from requests_toolbelt.threaded import pool
conn = sqlite3.connect('tiny_pipeline.db')

def get_data(symbol):
        
        history = yfinance.Ticker(symbol).history(period='7y').reset_index()
        
        history = get_ta(history, volume=True, pattern=False)
        history.columns = map(str.lower, history.columns)
        history['return'] = history['close'].pct_change() * 100
        history = history.dropna()
        history['next_return'] = history['return'].shift(-1)
        
        #num_rows = len(history)
        #train = history.head( int(num_rows * .75) )
        #test = history.tail( int(num_rows *.25) )
        history['symbol'] = symbol
        return history

def get_etfs():        
    
    finviz_url = 'https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund,ipodate_more10,sh_avgvol_o2000,sh_opt_optionshort&r=%s'
    
    
    finviz_page = requests.get(finviz_url % 1)
    ticker_count = int(re.findall('Total: </b>[0-9]*', finviz_page.text)[0].split('>')[1])
    urls = []

    for ticker_i in range(1, ticker_count, 20):
        urls.append(finviz_url % ticker_i)
        #break

    p = pool.Pool.from_urls(urls)
    p.join_all()

    total_etf_df = []
    for response in p.responses():
        start = response.text.find('<table width="100%" cellpadding="3" cellspacing="1" border="0" bgcolor="#d3d3d3">')
        end = response.text.find('</table>',start)+10

        #tickers = re.findall(r'primary">[A-Z]*', response.text)
        df = pd.read_html(response.text[start:end])[0]
        df.columns = df.loc[0]
        df = df.drop([0])
        total_etf_df.append(df)
    total_etf_df = pd.concat(total_etf_df)
    del total_etf_df['No.']    
    print(total_etf_df)
    return total_etf_df['Ticker'].values

if __name__ == '__main__':
    features = ['adxr', 'mfi', 'trix', 'trange', 'volume']
    #symbols = get_etfs()
    
    #symbols = ['FFIV', 'FLIR', 'FSLR', 'IRM', 'JNPR', 'NLSN', 'TRIP', 'XRX']
    #symbols = ['SPY','QQQ', 'DIA', 'QLD', 'SPUU']
    symbols = ['TQQQ','SPXL','GDX','EWZ','UPRO','USO','IBB','GDXJ','SDOW','SSO','SMH','RSX','VNQ','XLP','YINN','IYR','IWR','SPY','QQQ']
    total_results = []
    for s in symbols:
        symbols = [s, 'SPY']

        
        dfs = []
        for symbol in symbols:
            dfs.append(get_data(symbol))
        df = pd.concat(dfs)
        df = df.sort_values(by='date')
        
        num_rows = len(df)
        train = df.head( int(num_rows * .75) )
        test = df.tail( int(num_rows *.25) )
        print('running pipeline')
        x = pipeline(train, test, features, target_symbol = symbols[0])
        total_results.append(x.results)

        results = pd.concat(total_results)
        print(results)
        results.to_csv('test.csv')
        print(results.groupby(by='state_name').mean())
        #input()
    results = pd.concat(total_results)
    print(results)
    print(results.groupby(by='state_name').mean())
    input()