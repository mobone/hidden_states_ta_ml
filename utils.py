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
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from math import floor, ceil

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


def plot_results(model, test_data, model_name):
    sns.set(font_scale=1.25)
    style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,'legend.frameon': True}
    sns.set_style('white', style_kwds)

    colors = cm.rainbow(np.linspace(0, 1, model.n_components))
    
    #print(test_data['state'].values)
    #test_data.loc[test_data['state']==2, 'state'] = 1
    sns.set(font_scale=1.5)
    states = (pd.DataFrame(test_data['state'].values, columns=['states'], index=test_data.index)
            .join(test_data, how='inner')
            .reset_index(drop=False)
            .rename(columns={'index':'Date'}))
    #print(states.tail(100))

    #suppressing warnings because of some issues with the font package
    #in general, would not rec turning off warnings.
    import warnings
    warnings.filterwarnings("ignore")

    sns.set_style('white', style_kwds)
    order = [0, 1, 2]
    fg = sns.FacetGrid(data=states, hue='states', hue_order=order,
                    palette=colors, aspect=1.31, height=12)
    fg.map(plt.scatter, 'date', "close", alpha=0.8).add_legend()
    sns.despine(offset=10)
    fg.fig.suptitle('Historical SPY Regimes', fontsize=24, fontweight='demi')
    #plt.show()
    plt.savefig('./plots/%s.png' % model_name)

def walk_timeline(test_data, buy_state, good_state_num, middle_state_num, bad_state_num):
    # todo: use middle state for regular index and best state for bonus index
    tqqq_history = yfinance.Ticker('TQQQ').history(period='5y', auto_adjust=False)
    tqqq_history = tqqq_history.reset_index()
    #print(self.test.head(1).index.values[0])
    tqqq_history = tqqq_history[tqqq_history['Date']>=test_data.head(1).index.values[0]]
    tqqq_history = tqqq_history.iloc[:-1]
    
    total_results = []

    share_accumulation_rate = 3
    share_decay_rate = 1
    """
    for share_accumulation_rate in range(1,10):
        share_accumulation_rate = share_accumulation_rate * 5
        for share_decay_rate in range(1,10):
    """
    #share_decay_rate = share_decay_rate * 5
    #share_accumulation_rate = 3
    num_shares = 0
    #max_shares = 50
    
    keep_for_bonus = 2000
    balance = 8000

    full_count = share_accumulation_rate

    starting_balance = 8000 + keep_for_bonus

    balances = []

    held_shares = {}
    held_shares['QQQ'] = {'num_shares': 0}
    held_shares['TQQQ'] = {'num_shares': 0}
    """
    print(test_data.head(1))
    print(test_data.tail(1))
    print(tqqq_history.head(1))
    print(tqqq_history.tail(1))
    input()
    """
    
    if buy_state == 'good':
        buy_state = good_state_num
        bonus_buy_state = good_state_num
    elif buy_state == 'middle':
        
        buy_state = middle_state_num
        bonus_buy_state = good_state_num

    bad_state = bad_state_num
    buy_and_hold_start = balance + keep_for_bonus
    benchmark_return = balance * float(test_data['close'].tail(1)) / float(test_data['close'].head(1))
    tqqq_return = keep_for_bonus * float(tqqq_history['Close'].tail(1)) / float(tqqq_history['Close'].head(1))
    
    #test_data = test_data[   list(set(['close']+self.features+['state']))  ]
    for i in test_data.index:
        today = test_data.loc[i]
        today_bonus = tqqq_history.loc[tqqq_history['Date'] == i]
        bonus_share_price = float(today_bonus['Close'])
        share_price = float(today['close'])
        #print(pd.DataFrame(today).T)

        num_shares_all_funds = floor(balance / share_price)
        num_shares_some_funds = floor(balance / share_accumulation_rate / share_price)
        num_bonus_shares_to_buy = floor(keep_for_bonus / bonus_share_price)
        #print('possible num shares', num_shares_some_funds, num_shares_all_funds, full_count)
        if full_count:
            num_shares_to_buy = num_shares_some_funds
        else:
            num_shares_to_buy = num_shares_all_funds
        
        if (today['state']==buy_state) and num_shares_to_buy > 0 and (balance - (num_shares_to_buy*share_price) > 0): # and held_shares['QQQ']['num_shares']<max_shares:
            # buy more shares if we can
            #print('buying %s shares at %s for a cost of %s' % ( num_shares_to_buy, share_price, round(num_shares_to_buy * share_price, 2)))
            held_shares['QQQ']['num_shares'] = held_shares['QQQ']['num_shares'] + num_shares_to_buy
            balance = balance - round( (num_shares_to_buy*share_price), 2)
        
        if (today['state']==bonus_buy_state) and full_count == 0 and held_shares['TQQQ']['num_shares']==0:
            # use bonus if we can't buy more regular shares
            #print('buying %s bonus shares at %s for a cost of %s' % ( num_bonus_shares_to_buy, bonus_share_price, round(num_bonus_shares_to_buy * bonus_share_price, 2)))
            #print(keep_for_bonus)
            #print(bonus_share_price)
            bonus_shares = floor(keep_for_bonus / bonus_share_price)
            held_shares['TQQQ']['num_shares'] = bonus_shares
            keep_for_bonus = keep_for_bonus - (bonus_shares * bonus_share_price)

        if today['state']==bad_state:
            full_count = share_accumulation_rate + 1
            #num_shares_after_selling = ceil(held_shares['QQQ']['num_shares']/share_decay_rate)
            #num_shares_sold = held_shares['QQQ']['num_shares'] - num_shares_after_selling
            #held_shares['QQQ']['num_shares'] = num_shares_after_selling
            num_shares_sold = (held_shares['QQQ']['num_shares'] / share_decay_rate)
            num_shares_after_selling = held_shares['QQQ']['num_shares'] - num_shares_sold
            #print('selling %s shares for a value of %s. now down to %s shares' % ( num_shares_sold, round(num_shares_sold*share_price,2), num_shares_after_selling ))
            balance = balance + round( (num_shares_sold * share_price), 2)
            held_shares['QQQ']['num_shares'] = num_shares_after_selling
            
            

            if held_shares['TQQQ']['num_shares']!=0:
                keep_for_bonus = keep_for_bonus + (held_shares['TQQQ']['num_shares'] * bonus_share_price)
                
                #print('selling %s bonus shares for a value of %s. now down to %s shares' % ( held_shares['TQQQ']['num_shares'], round((held_shares['TQQQ']['num_shares'] * bonus_share_price),2), 0 ))
                held_shares['TQQQ']['num_shares'] = 0
        full_count = full_count - 1
        if full_count < 0:

            full_count = 0
            

        

        held_balance = 0
        #for key in held_shares.keys():
        held_balance = held_balance + round( (held_shares['QQQ']['num_shares'] * float(share_price)), 2 )
        held_balance = held_balance + round(  held_shares['TQQQ']['num_shares'] * float(bonus_share_price), 2)
        balances.append( [today['state'], balance, keep_for_bonus, held_balance, balance + keep_for_bonus + held_balance, held_shares['QQQ']['num_shares'], held_shares['TQQQ']['num_shares']] )
        #print(pd.DataFrame(balances, columns = ['state', 'bank', 'bonus', 'held_balance', 'total', 'held_shares', 'held_bonus_shares']))
        #input()
        
        
    
    ending_balance = balance + keep_for_bonus + round( (held_shares['QQQ']['num_shares'] * share_price), 2) + round( (held_shares['TQQQ']['num_shares'] * bonus_share_price), 2)
    #print(ending_balance)
    buy_and_hold_percent = (benchmark_return + tqqq_return) / buy_and_hold_start - 1
    
    total_results.append([share_accumulation_rate, share_decay_rate, round(ending_balance, 2), round((ending_balance / starting_balance - 1),4), round(buy_and_hold_percent, 4)])
    result_df = pd.DataFrame(total_results, columns = ['accum','decay', 'ending balance', 'percent', 'benchmark'])
    #print(result_df)
    #print(result_df['percent'].max())
    percent_return = result_df['percent'].max()
    benchmark_return = round(buy_and_hold_percent,4)
    #result_df.to_csv('test.csv')
            
    return percent_return, benchmark_return

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