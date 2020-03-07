import pandas as pd
import sqlite3
import warnings
import yfinance
from math import floor
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

class trader():
    def __init__(self, data, regular_symbol, strong_symbol, short_symbol = None):
        #print(data)
        self.df = data[['date','close','state_name']]
        self.df = self.df.reset_index(drop=True)

        
        self.short_symbol = short_symbol
        self.regular_symbol = regular_symbol
        self.strong_symbol = strong_symbol

        self.stock_history = {}

        if self.short_symbol is not None:
            self.get_stock_data(self.short_symbol)

        self.get_stock_data(self.regular_symbol)
        self.get_stock_data(self.strong_symbol)
        
        self.df['date'] = pd.to_datetime(self.df['date'])

        self.held_shares = {}
        
        self.starting_balance = 10000
        self.bank_balance = self.starting_balance
        
        self.run_trades()


    def get_stock_data(self, symbol):
        x = yfinance.Ticker(symbol).history(period='10y', auto_adjust=False)
        x = x.reset_index()
        x.columns = map(str.lower, x.columns)
        x['date'] = pd.to_datetime(x['date'])
        self.stock_history[symbol] = x

    def run_trades(self):
        num_trades = 0
        list_of_trades = []
        last_date = self.df['date'].tail(1).values[0]
        for day_index in range(len(self.df)):
            
            day = self.df.loc[day_index, ['date', 'state_name', 'close']]
            #print(day)
            state_name = day['state_name']
            #self.share_price = float(day['close'])
            self.get_todays_prices(day)
            try:
                self.liquidate()
            except Exception as e:
                #print(e)
                pass
                
            
            if day['date']==last_date:
                break
            
            list_of_trades.append( [ day['date'], self.bank_balance ] )
            #print(pd.DataFrame(list_of_trades))
            
            
            # make trades
            if self.short_symbol is not None and state_name == 'sell':
                self.buy_shares( self.short_symbol )
            elif state_name == 'buy':
                self.buy_shares( self.regular_symbol )
            elif state_name == 'strong_buy':
                self.buy_shares( self.strong_symbol )
            
        percent_change = round(self.bank_balance / self.starting_balance - 1, 4)*100
        df = pd.DataFrame(list_of_trades, columns = ['date','balance'])
        #df.plot.line(x='date',y='balance')
        #plt.show()
        #print('%s %s' % (  self.bank_balance, percent_change ) )
        self.return_percentage = percent_change
        self.num_trades = num_trades
        


    def liquidate(self):
        
        if self.short_symbol is not None:
            held_balance = round( (self.held_shares[self.short_symbol]['num_shares']*self.short_price), 2)
            self.bank_balance = round(self.bank_balance + held_balance, 2)
        held_balance = round( (self.held_shares[self.regular_symbol]['num_shares']*self.regular_price), 2)
        self.bank_balance = round(self.bank_balance + held_balance, 2)
        held_balance = round( (self.held_shares[self.strong_symbol]['num_shares']*self.strong_price), 2)
        self.bank_balance = round(self.bank_balance + held_balance, 2)

        if self.short_symbol is not None:
            del self.held_shares[self.short_symbol]
        del self.held_shares[self.regular_symbol]
        del self.held_shares[self.strong_symbol]

    def get_todays_prices(self, day):
        if self.short_symbol is not None:
            history = self.stock_history[self.short_symbol]
            self.short_price = float(history[history['date']==day['date']]['close'])

        history = self.stock_history[self.regular_symbol]
        self.regular_price = float(history[history['date']==day['date']]['close'])

        history = self.stock_history[self.strong_symbol]
        self.strong_price = float(history[history['date']==day['date']]['close'])


    def buy_shares(self, symbol):
        if self.short_symbol is not None and symbol == self.short_symbol:
            percent_used = (0.5, 0, 0)
        elif symbol == self.regular_symbol:
            percent_used = (0, .9, .1)
        else:
            percent_used = (0, 0, 1)

        if self.short_symbol is not None:
            num_shares_short =  floor( (self.bank_balance * percent_used[0]) / self.short_price )
        num_shares_regular =  floor( (self.bank_balance * percent_used[1]) / self.regular_price )
        num_shares_strong =  floor( (self.bank_balance * percent_used[2]) / self.strong_price )

        if self.short_symbol is not None:
            self.bank_balance = self.bank_balance - round(num_shares_short * self.short_price, 2)
        self.bank_balance = self.bank_balance - round(num_shares_regular * self.regular_price, 2)
        self.bank_balance = self.bank_balance - round(num_shares_strong * self.strong_price, 2)

        if self.short_symbol is not None:
            self.held_shares[self.short_symbol] = {'num_shares': num_shares_short}
        self.held_shares[self.regular_symbol] = {'num_shares': num_shares_regular}
        self.held_shares[self.strong_symbol] = {'num_shares': num_shares_strong}
        #print(self.held_shares)
        #input()
        
        
        
def trade_runner(model_name):
    trader(model_name)

if __name__ == "__main__":
    conn = sqlite3.connect('hmm.db')
    sql = 'select name from models_with_test order by real_test_correl desc'
    model_names = list(pd.read_sql(sql, conn)['name'].values)
    print(model_names)
    
    p = Pool( int(cpu_count()-1) )
    p.map(trade_runner, model_names)
    #trader(model_names[0])