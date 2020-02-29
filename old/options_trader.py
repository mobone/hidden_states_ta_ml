import pandas as pd
from pymongo import MongoClient
import sqlite3
from datetime import timedelta
import warnings
from math import floor
from mongo_memoize import memoize
warnings.simplefilter("ignore")
model_name = 'tasty-black-forest'

conn = sqlite3.connect('hidden_states.db')
client = MongoClient('192.168.1.142')
db = client['alpha']
collection = db['options']



class options_trader():
    def __init__(self, symbol, trades):
        self.symbol = symbol
        self.trades = trades
        self.results = []
        for _, trade in self.trades.iterrows():
            self.stock_buy_date = pd.to_datetime(trade['buy_date'])
            
            self.stock_sell_date = pd.to_datetime(trade['sell_date'])+timedelta(hours=18)

            try:
                
                self.raw_options = self.get_options()
                
                self.parse_options()
            
                self.get_start_option()
                self.get_highest_gain_option()
                self.get_end_option()
                result = self.aggregate_results()
                self.results.append( [self.stock_buy_date, self.option_buy_price, self.final_profit_percent, self.profit_dollars] )
            except Exception as e:
                #print(e)
                #input()
                pass


        
        
    def parse_options(self):
        options = self.raw_options
        options_df = pd.DataFrame.from_dict(options)
        
        if options_df.empty:
            raise
        
        options_df = options_df.sort_values(by=['update_hour', 'expiration'])
        
        

        options_df['update_hour'] = options_df['update_hour'].astype(int)

        options_df = options_df[options_df['update_hour']>=9]
        
        self.options_df = options_df
        

    def get_start_option(self):
        options_df = self.options_df

        # filter out ask's that are less than $250
        options_df = options_df[options_df['ask']>2.5]

        # only select if actually morning
        options_df = options_df[options_df['update_hour']<=10]
        if options_df.empty:
            raise
        
        # select nearest expiration
        self.expiration = options_df['expiration'].unique()[0]
        options_df = options_df[options_df['expiration']==self.expiration]

        # select options that are under $2k
        options_df = options_df[options_df['ask']<=20]

        # select the morning option
        start_option = options_df[options_df['update_hour']==options_df['update_hour'].min()]
        
        # select the most popular option
        self.start_option = start_option[start_option['volume']==start_option['volume'].max()]
        
        self.start_option['trade'] = 'start'

        self.contract_symbol = self.start_option['contractsymbol'].values[0]
        self.option_buy_price = self.start_option['ask'].values[0]
        
        

    def get_highest_gain_option(self):
        options_df = self.options_df

        options_df = options_df[options_df['contractsymbol']==self.contract_symbol]

        options_df = options_df[options_df['bid']==options_df['bid'].max()]

        self.highest_option = options_df

        self.highest_option['trade'] = 'high'


    def get_end_option(self):
        options_df = self.options_df

        # get all entries for given contract symbol
        options_df = options_df[options_df['contractsymbol']==self.contract_symbol]

        options_df = options_df[options_df['update_hour']>=12]
        if options_df.empty:
            raise

        options_df = options_df[options_df['update_hour']==14]

        self.end_option = options_df

        self.end_option['trade'] = 'end'
        
        

    def aggregate_results(self):
        traded_option = pd.concat([self.start_option, self.highest_option, self.end_option])
        
        start_price = float(traded_option.loc[traded_option['trade']=='start', 'ask'])

        high_price = float(traded_option.loc[traded_option['trade']=='high', 'bid'])
        
        end_price = float(traded_option.loc[traded_option['trade']=='end', 'bid'])

        self.final_profit_percent = round(end_price / start_price - 1, 4)
        self.max_profit = round(high_price / start_price - 1, 4)

        num_contracts = floor(2000.0/ (start_price*100))
        self.profit_dollars = num_contracts * start_price * self.final_profit_percent * 100
        #print(self.stock_buy_date, final_profit, max_profit, start_price, high_price, end_price)

    
    def get_options(self):
        query = {'ticker': 'SPY', 'type': 'Call', 'update_datetime': {'$gt': self.stock_buy_date, '$lt': self.stock_sell_date}}
        return list(collection.find(query))
        
sql = 'select name, features, accuracy from "model_summaries_indexes" where num_trades > 20 and accuracy > .7 and trade_time == "intraday" order by accuracy DESC'
models = pd.read_sql(sql, conn)
print(models)
model_names = list(models.drop_duplicates(subset=['features'])['name'])
print(model_names)


for model_name in model_names:
    print(model_name)
    sql = 'select * from trades_indexes where name == "%s"' % model_name
    df = pd.read_sql(sql, conn)
    df = df[df['ticker']=='SPY']
    for symbol, trades in df.groupby(by='ticker'):
        x = options_trader(symbol, trades)
        options_traded = pd.DataFrame(x.results, columns = ['trade_date', 'option_price', 'profit_percent', 'profit_dollars'] )
        
        
        #print(options_traded)
        
        print(pd.DataFrame(options_traded['profit_dollars'].describe()).T)
        print(options_traded['profit_dollars'].sum())
        print()
        