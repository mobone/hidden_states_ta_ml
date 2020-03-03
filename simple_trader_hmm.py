import pandas as pd
import sqlite3
import warnings
import yfinance
from math import floor
warnings.simplefilter('ignore')

class trader():
    def __init__(self, model_name):
        conn = sqlite3.connect('hmm.db')
        sql = 'select * from trades where name ==  "%s"' % model_name
        self.df = pd.read_sql(sql, conn)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.regular_symbol = 'QLD'
        self.strong_symbol = 'TQQQ'

        self.stock_history = {}
        self.held_shares = {}
        

        self.get_stock_data(self.regular_symbol)
        self.get_stock_data(self.strong_symbol)

        self.bank_balance = 10000
        self.run_trades()

    def get_stock_data(self, symbol):
        x = yfinance.Ticker(symbol).history(period='5y', auto_adjust=False)
        x = x.reset_index()
        x.columns = map(str.lower, x.columns)
        self.stock_history[symbol] = x


    def run_trades(self):
        list_of_trades = []
        last_date = self.df['date'].tail(1).values[0]
        for day_index in range(10,len(self.df)):
            days = self.df.loc[:day_index, ['date', 'state', 'close']].tail(10)
            day = days.tail(1)
            print()
            print('last 10 days\n', days)
            print('today\n', day)
            print()
            
            
            # update the bank balance from yesterdays trades
            for symbol in self.held_shares.keys():
                self.rebalance(symbol, day)
            self.bank_balance = round(self.bank_balance, 2)
            
            if day['date'].values[0]==last_date:
                print('finished')
                break
            print(self.bank_balance)
            list_of_trades.append( [ day['date'].values[0], self.bank_balance ] )
            # find out if we need to sell everything
            if not day[day['state']==0.0].empty:
                print('bad day! not buying')
                #input()
                continue

            # find out the percentages for number of shares to buy
            num_regular = days[days['state']==1.0]['state'].count()
            num_strong = days[days['state']==2.0]['state'].count()

            regular_percent = num_regular / sum([num_regular, num_strong])
            strong_percent = num_strong / sum([num_regular, num_strong])
            if regular_percent>0:
                self.buy_shares(self.regular_symbol, day, regular_percent)
            if strong_percent>0:
                self.buy_shares(self.strong_symbol, day, strong_percent)
        print(self.bank_balance)
        df = pd.DataFrame(list_of_trades, columns = ['date', 'balance'])
        df.to_csv('qld.csv')
            
            
            
    def rebalance(self, symbol, day):
        this_history = self.stock_history[symbol]
        
        this_history = this_history[this_history['date']==day['date'].values[0]]
        print(this_history)
        share_price = float(this_history['close'])
        num_shares = self.held_shares[symbol]['num_shares']
        if num_shares == 0:
            return
        self.held_shares[symbol]['num_shares'] = 0
        self.bank_balance = self.bank_balance + round( (num_shares*share_price), 2)
        
        print('rebalancing, sold %s shares of %s at $%s a share' % ( num_shares, symbol, share_price ))


    def sell_all(self):
        print('selling everything')
        #input()

    def buy_shares(self, symbol, day, percent):
        this_history = self.stock_history[symbol]
        
        this_history = this_history[this_history['date']==day['date'].values[0]]
        share_price = float(this_history['close'])
        num_shares =  floor( (self.bank_balance * percent) / share_price )
        if num_shares == 0:
            return
        self.held_shares[symbol] = {'num_shares': num_shares}
        self.bank_balance = self.bank_balance - round(num_shares * share_price, 2)
        print('bought %s shares of %s at $%s per share' % ( num_shares, symbol, share_price) )
        #if symbol == self.strong_symbol:
        #    input()
        
        
            


trader("homey-linen-ray")