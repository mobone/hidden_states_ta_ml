import pandas as pd
import sqlite3
import warnings
import yfinance
from math import floor
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

class trader():
    def __init__(self, target = None, model_name=None, data=None):
        if model_name is None and data is not None:
            self.df = data[['date','close','state_name']]
            self.df = self.df.reset_index(drop=True)
        elif model_name is not None and data is None:
            self.model_name = model_name
            conn = sqlite3.connect('hmm_rolling.db')
            sql = 'select * from trades_final where name ==  "%s"' % model_name
            self.df = pd.read_sql(sql, conn)
            self.df = self.df.reset_index(drop=True)
        elif data is None and model_name is None:
            print('need data or model name')
            raise

        if target == None:
            self.target = ['strong_buy']
        else:
            self.target = target

        
        
        self.df['date'] = pd.to_datetime(self.df['date'])

        self.num_held = 0

        
        
        self.starting_balance = 10000
        self.bank_balance = self.starting_balance
        
        self.run_trades()


    def run_trades(self):
        num_trades = 0
        list_of_trades = []
        last_date = self.df['date'].tail(1).values[0]
        for day_index in range(10,len(self.df)):
            day = self.df.loc[:day_index, ['date', 'state_name', 'close']].tail(1)

            self.share_price = float(day['close'])
            
            self.liquidate()
            
            if day['date'].values[0]==last_date:
                break
            list_of_trades.append( [ day['date'].values[0], self.bank_balance ] )
            
            # find out if we need to sell everything
            if (day['state_name'] == 'sell').bool():
                continue
            elif  (day['state_name'].isin(self.target)).bool():
                self.buy_shares(day)
                num_trades += 1
            
        percent_change = round(self.bank_balance / self.starting_balance - 1, 4)*100
        df = pd.DataFrame(list_of_trades, columns = ['date','balance'])
        #df.plot.line(x='date',y='balance')
        #plt.show()
        #print('%s %s' % (  self.bank_balance, percent_change ) )
        self.return_percentage = percent_change
        self.num_trades = num_trades
        


    def liquidate(self):
        held_balance = round( (self.num_held*self.share_price), 2)
        self.bank_balance = round(self.bank_balance + held_balance, 2)
        self.num_held = 0


    def buy_shares(self, day):
        
        if ( day['state_name']=='buy' ).bool():
            percent = 0.5
        elif ( day['state_name']=='strong_buy' ).bool():
            percent = 1.00
        num_shares =  floor( (self.bank_balance * percent) / self.share_price )

        
        self.num_held = num_shares
        self.bank_balance = self.bank_balance - round(num_shares * self.share_price, 2)
        
        
        
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