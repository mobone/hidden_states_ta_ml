import pandas as pd
import sqlite3
import yfinance
from math import floor
from hidden_states_multiprocess import get_industry_tickers
conn = sqlite3.connect('hidden_states.db')

sql = 'select name from hidden_states_models_technology where num_trades>10 order by abnormal_mean desc'
model_names = pd.read_sql(sql, conn)['name']

#model_names = ['seedy-silver-bonobo']

sector = 'technology'

tickers = get_industry_tickers(sector)
total_num_tickers = len(tickers)

tickers = tickers[:int(total_num_tickers/2)]
tickers = tickers + ['QQQ']
print(tickers)

histories = {}
for ticker in tickers:
    history = yfinance.Ticker(ticker).history(period='2y').reset_index()
    history = history[ (history['Date']>'2019-01-01') & (history['Date']<'2019-12-31')]
    history['ticker'] = ticker
    histories[ticker] = history

model_results = []
for model_name in model_names:
    
    sql = 'select * from hidden_states_models_trades_technology where name == "%s"' % model_name
    trades = pd.read_sql(sql, conn)

    trades['buy_date'] = pd.to_datetime(trades['buy_date'])
    trades['sell_date'] = pd.to_datetime(trades['sell_date'])

    #print(trades)

    start_balance = 20000.0
    money_used = 1000
    balance = start_balance
    held_trades = {}
    simulation_results = []
    
    
    # buy the index as well
    num_shares = floor( (start_balance*.2) / float(histories['QQQ']['Open'].head(1)) )
    #num_shares = 1
    held_trades['QQQ'] = {'buy_date': pd.to_datetime('2019-01-02'),
                         'sell_date': None,
                         'buy_price': float(histories['QQQ']['Open'].head(1)),
                         'sell_price': float(histories['QQQ']['Close'].tail(1)),
                         'num_shares': num_shares}
    balance = balance - (num_shares * float(histories['QQQ']['Open'].head(1)))
    
    
    #print(histories['QQQ'])
    #input()
    trading_change = float(histories['QQQ']['Close'].tail(1))  / float(histories['QQQ']['Open'].head(1)) - 1
    try:
        for trade_date in histories['QQQ']['Date']:
            #print(trade_date)

            # sell trades
            for ticker in list(held_trades.keys()):
                this_held_trade = held_trades[ticker]
                if trade_date == this_held_trade['sell_date']:
                    
                    balance = balance + ( this_held_trade['num_shares'] * this_held_trade['sell_price'] )
                    del held_trades[ticker]

            # make new trades
            if not trades[trades['buy_date']==trade_date].empty:
                
                for key, possible_trade in trades[trades['buy_date']==trade_date].iterrows():
                    #print('possible trade')
                    #print(possible_trade)
                    #print('--')
                    if possible_trade['ticker'] not in held_trades.keys():
                        # find out how much to use for the trade
                        #money_used = min( balance * .15, 500)

                        #num_shares = floor( (balance*.15) / possible_trade['buy_price'] )
                        num_shares = floor( money_used / possible_trade['buy_price'] )
                        if balance - (num_shares * possible_trade['buy_price']) < 0:
                            continue
                        held_trades[possible_trade['ticker']] = {'buy_date': possible_trade['buy_date'],
                                                            'sell_date': possible_trade['sell_date'],
                                                            'buy_price': possible_trade['buy_price'],
                                                            'sell_price': possible_trade['sell_price'],
                                                            'num_shares': num_shares}
                        balance = balance - (num_shares * possible_trade['buy_price'])
            
            
            held_balance = 0
            # lookup the account balance
            for ticker in held_trades.keys():
                stock_history = histories[ticker]
                current_share_price = stock_history[stock_history['Date']==trade_date]['Close'].values[0]
                num_shares = held_trades[ticker]['num_shares']
                held_balance += ( current_share_price * num_shares )

            simulation_results.append( [trade_date, balance, held_balance, balance + held_balance, len(held_trades.keys())] )
            
            #print(len(held_trades.keys()))
            
            #print(pd.DataFrame(simulation_results, columns=['date', 'bank_balance', 'held_balance', 'total_balance', 'num_held']))
            
            


        df = pd.DataFrame(simulation_results, columns=['date', 'bank_balance', 'held_balance', 'total_balance', 'num_held'])
        #print(df)
        #print(df['num_held'].max())
        model_results.append( [model_name, df['bank_balance'].min(), df['num_held'].max(), df['total_balance'].tail(1).values[0]/start_balance - 1, trading_change ])

        df = pd.DataFrame(model_results, columns = ['name', 'min_bank_balance', 'max_held', 'return', 'benchmark_return'])
        print(df.sort_values(by=['return']))
        
        benchmark_return = df['benchmark_return'].values[0]
        print('======')
        print( df[ df['return'] >= benchmark_return].sort_values(by=['return']) )
        print('======')
    except:
        continue