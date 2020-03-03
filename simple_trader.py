import pandas as pd
import yfinance
import sqlite3
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import sleep
from random import shuffle
from utils import plot_results

# TODO: strip down and add bonus; use only buy for regular, and strong buy for bonus
class trader():
    def __init__(self, model_name, accum_rate, decay_rate, trade_states, index):
        self.index = index
        self.model_name = model_name

        self.bank_value = 10000
        
        self.accum_rate = accum_rate
        self.decay_rate = decay_rate
        self.trade_states = trade_states

        self.current_accum = self.accum_rate
        self.held_shares = {}
        self.held_shares['TQQQ'] = {'num_shares': 0}

        conn = sqlite3.connect('markov_models.db')
        sql = 'select * from trades where name = "%s"' % model_name
        self.trade_days = pd.read_sql(sql, conn)
        print(self.trade_days)
        plot_results(self.trade_days, model_name)
        return


        # get TQQQ
        self.tqqq = yfinance.Ticker('TQQQ').history(period='5y', auto_adjust=False).reset_index()
        self.tqqq.columns = map(str.lower, self.tqqq.columns)

        self.qqq = yfinance.Ticker('QQQ').history(period='5y', auto_adjust=False).reset_index()
        self.qqq.columns = map(str.lower, self.qqq.columns)


        # get TQQQ performance
        
        self.tqqq = self.tqqq[ (self.tqqq['date']>=self.trade_days.head(1)['date'].values[0]) & (self.tqqq['date']<=self.trade_days.tail(1)['date'].values[0]) ]
        self.tqqq_start_price = float(self.tqqq.head(1)['close'])
        self.tqqq_performance = float(self.tqqq.tail(1)['close']) / float(self.tqqq.head(1)['close']) 

        self.qqq = self.qqq[ (self.qqq['date']>=self.trade_days.head(1)['date'].values[0]) & (self.qqq['date']<=self.trade_days.tail(1)['date'].values[0]) ]
        self.qqq_start_price = float(self.qqq.head(1)['close'])
        
        
        
        # start trading
        self.run_trades()
        

    def run_trades(self):
        completed_trades = []
        consecutive_days = 0
        for _, self.day in self.trade_days.iterrows():
            self.tqqq_day = self.tqqq[self.tqqq['date']==self.day['date']]
            self.qqq_day = self.qqq[self.qqq['date']==self.day['date']]

            #print(self.day.to_frame())
            #print(self.tqqq_day)

            # check if state is buy 
            #print(self.day['state'], self.current_accum)
            """
            if self.day['state'] in self.trade_states:
                consecutive_days += 1
            else:
                consecutive_days = 0
            """ 
            if self.day['state'] in self.trade_states and self.current_accum!=0:
            #if consecutive_days>=4 and self.day['state'] in self.trade_states and self.current_accum!=0:
                self.buy_shares()
                

            if self.day['state'] == 'sell' and self.held_shares['TQQQ']['num_shares']>0:
                self.sell_shares()
                
                
            held_balance = self.held_shares['TQQQ']['num_shares'] * float(self.tqqq_day['close'])
            completed_trades.append( [ 
                                        self.tqqq_day['date'].values[0],
                                        self.bank_value, 
                                        held_balance, 
                                        self.bank_value + held_balance, 
                                        round( float(self.bank_value + held_balance) / 10000.0 - 1, 4),
                                        round( float(self.qqq_day['close']) / self.qqq_start_price - 1, 4),
                                    ])
        df = pd.DataFrame( completed_trades, columns = ['date', 'bank', 'held', 'total', 'percent_change', 'index_percent_change'] ) 
        print(df)
        #if df[df['percent_change']<df['index_percent_change']].empty:
        df = df.set_index('date')
        df[ ['percent_change', 'index_percent_change'] ].plot(style=['o','rx'])
        #stddev = (df['percent_change']-df['index_percent_change']).std()
        #changes = df['percent_change'].pct_change() 
        #changes = changes.replace([np.inf, -np.inf], np.nan)
        #changes = changes.dropna()
        
        #stddev = changes.std()
        stddev = (df['percent_change'] - df['index_percent_change']).std()
        #stddev = df['percent_change'].std()
        filename = '%s_%s_%s_%s.png' % ( round(stddev, 2), self.index, self.accum_rate, self.decay_rate )
        print(filename)
        plt.savefig('./plots/%s' % filename)
        plt.close()
        
        """
        if float(df.tail(1)['percent_change']) > float(df.tail(1)['index_percent_change']):
            df.to_csv('test.csv')
            print('found one!')
            input()
        """
        

        

    def buy_shares(self):
        share_price = float(self.tqqq_day['close'])
        num_shares = floor(self.bank_value / self.current_accum / share_price)

        self.held_shares['TQQQ']['num_shares'] += num_shares

        self.bank_value = self.bank_value - (num_shares * share_price)

        self.current_accum = self.current_accum - 1
        #print('bought shares')

    def sell_shares(self):
        
        share_price = float(self.tqqq_day['close'])
        
        num_shares_to_sell = ceil(self.held_shares['TQQQ']['num_shares']/self.decay_rate)
        
        self.held_shares['TQQQ']['num_shares'] -= num_shares_to_sell
        
        #num_shares_to_sell = self.held_shares['TQQQ']['num_shares']

        #self.held_shares['TQQQ']['num_shares'] = 0

        self.bank_value = self.bank_value + (num_shares_to_sell * share_price)

        self.current_accum = self.current_accum + 1
        if self.current_accum > self.accum_rate:
            self.current_accum = self.accum_rate
        #print('sold shares')



def run_trader(params):
    model_name, accum_rate, decay_rate, trade_states, index = params

        
    trader(model_name, accum_rate, decay_rate, trade_states, index)
if __name__ == '__main__':
    params_list = []
    
    conn = sqlite3.connect('markov_models.db')
    sql = 'select name, avg(good_mean) from models_multi_year group by name order by avg(good_mean) desc limit 30'

    models = pd.read_sql(sql, conn)
    
    for index, model in models.iterrows():
        for accum_rate in range(1,5):
            for decay_rate in range(1, 5):
                #for trade_states in [ ['strong buy'], ['buy'], ['strong buy', 'buy'] ]:
                for trade_states in [ ['strong buy']]:
                    #trader(model_name, accum_rate, decay_rate, trade_states)

                        params_list.append( [model['name'], accum_rate * 2 , decay_rate * 2, trade_states, index] )
    shuffle(params_list)

    p = Pool(16)
    p.map(run_trader, params_list)
    #run_trader(params_list[9])
    #p = Pool(15)
    #p.map(run_model, param_list)