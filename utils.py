from hidden_states_multiprocess import stock_predictor
import sqlite3
import pandas as pd
import requests as r
import re
from requests_toolbelt.threaded import pool
from multiprocessing import Pool
import time
from random import choices
from ta_indicators import get_ta
import requests
import yfinance
import numpy as np
def get_market_cap(market_cap):
    
    if 'K' in market_cap:
        market_cap = float(market_cap[:-1])*1000
    elif 'M' in market_cap:
        market_cap = float(market_cap[:-1])*1000000
    elif 'B' in market_cap:
        market_cap = float(market_cap[:-1])*1000000000
    return market_cap


def get_industry_tickers(sector):
    
    finviz_url = 'https://finviz.com/screener.ashx?v=111&f=exch_nasd,ipodate_more5,sec_technology,sh_avgvol_o1000,sh_opt_optionshort&o=-marketcap'
    finviz_url = finviz_url + '&r=%s'
        
    
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

        for key, this_row in df.iterrows():
            cap = get_market_cap(this_row['Market Cap'])
            df.loc[key, 'Numerical Market Cap'] = cap
        
        total_etf_df.append(df)
    total_etf_df = pd.concat(total_etf_df)
    total_etf_df = total_etf_df.sort_values(by=['Numerical Market Cap'], ascending=False)
    #print(total_etf_df)
    del total_etf_df['No.']

    return list(total_etf_df['Ticker'])
    

def get_stock_tickers(asset_type='etf'):
    if asset_type == 'etf' or asset_type=='both':
        
        #finviz_url = 'https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund,ipodate_more1,sh_avgvol_o100,sh_opt_option&r=%s'
        finviz_url = 'https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund,ipodate_more5,sh_avgvol_o1000,sh_opt_optionshort&r=%s'
        
        
        finviz_page = r.get(finviz_url % 1)
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
        
    if asset_type=='stock' or asset_type=='both':
        finviz_url = 'https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund,sh_avgvol_o50,sh_opt_option&r=%s'
        finviz_page = r.get(finviz_url % 1)
        ticker_count = int(re.findall('Total: </b>[0-9]*', finviz_page.text)[0].split('>')[1])
        urls = []

        for ticker_i in range(1, ticker_count, 20):
            urls.append(finviz_url % ticker_i)
            #break

        p = pool.Pool.from_urls(urls)
        p.join_all()

        total_stock_df = []
        for response in p.responses():
            start = response.text.find('<table width="100%" cellpadding="3" cellspacing="1" border="0" bgcolor="#d3d3d3">')
            end = response.text.find('</table>',start)+10

            #tickers = re.findall(r'primary">[A-Z]*', response.text)
            df = pd.read_html(response.text[start:end])[0]
            df.columns = df.loc[0]
            df = df.drop([0])
            total_stock_df.append(df)
        total_stock_df = pd.concat(total_stock_df)
        del total_stock_df['No.']

    
    tickers = []
    if asset_type=='etf':
        for ticker in list(total_etf_df['Ticker']):
            tickers.append(ticker)
    elif asset_type=='stock':
        for ticker in list(total_stock_df['Ticker']):
            tickers.append(ticker)
    if asset_type == 'both':
        for ticker in list(total_etf_df['Ticker']):
            tickers.append(ticker)
        for ticker in list(total_stock_df['Ticker']):
            tickers.append(ticker)

    return tickers

def test_stock(ticker_with_name):
    ticker, name = ticker_with_name
    conn = sqlite3.connect('hidden_states.db')
    sql = 'select * from hidden_states_test where name=="%s" limit 1' % name
    params = pd.read_sql(sql, conn)
    params[params.columns] = params[params.columns].apply(pd.to_numeric, errors='ignore')
    params = params.to_dict(orient='records')[0]
    params['ticker'] = ticker
    params['with_original'] = True
    
    print(params)
    x = stock_predictor(params)
    
    return ([ticker, name, x.num_trades, x.num_trades_profitable, x.total_return, x.accuracy], x.trades)

def run_test(ticker, name):
    tickers_with_symbols = []
    for ticker in tickers:
        tickers_with_symbols.append((ticker, name))
    start_time = time.time()
    p = Pool(16)
    results = p.map(test_stock, tickers_with_symbols )

    results_list = []
    for i in results:
        result_metrics, trades = i

        results_list.append(result_metrics)
        
        if trades is not None:
            trades.to_sql('trades', conn, if_exists='append')
        

    end_time = time.time()
    
    df = pd.DataFrame(results_list, columns=['ticker', 'name', 'num_trades', 'num_profitable_trades', 'total_return', 'accuracy'])
    df.to_sql('result_metrics', conn, if_exists='append')
    print(df)
    print(df.describe())


def get_data(tickers, period='5y', pattern=True):
    all_historic_data = []
    
    for ticker in tickers:
        #print('getting data for', ticker)
        ticker_data = yfinance.Ticker(ticker)
        ticker_data = ticker_data.history(period=period, auto_adjust=False)
        
        ticker_data = get_ta(ticker_data, True, pattern)
        ticker_data = ticker_data.reset_index()
        ticker_data.columns = map(str.lower, ticker_data.columns)

        ticker_data["return"] = ticker_data["close"].pct_change()
        ticker_data["range"] = (ticker_data["high"]/ticker_data["low"])-1
        ticker_data = ticker_data.drop(columns=['dividends','stock splits'])

        ticker_data["ticker"] = ticker
        
        ticker_data.dropna(how="any", inplace=True)
        ticker_data = ticker_data.reset_index(drop=True)

        all_historic_data.append(ticker_data)
    
    history_df = pd.concat(all_historic_data)
    #print(history_df)
    history_df = history_df.dropna(thresh=100,axis=1)
    
    
    
    history_df = history_df.replace([np.inf, -np.inf], np.nan)

    history_df = history_df.dropna()
    history_df = history_df.sort_values(by=['date'])
    history_df = history_df.reset_index(drop=True)

    return history_df
    

"""
if __name__ == '__main__':
    conn = sqlite3.connect('hidden_states.db')
    #tickers = get_stock_tickers()
    tickers = ['SPY', 'QQQ', 'TQQQ', 'QLD']
    sql = "select name from hidden_states_test order by total_return desc"
    model_names = list(pd.read_sql(sql, conn)['name'].unique())
    for model_name in model_names[:2]:
        print(model_name)
        params = {'name': model_name, 'tickers': tickers}
        #run_test(tickers, model_name)
        stock_predictor(params)
"""