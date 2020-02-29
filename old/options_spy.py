import pandas as pd
from pymongo import MongoClient
import sqlite3

conn = sqlite3.connect('hidden_states_next_day.db')
client = MongoClient('192.168.1.142')
db = client['finance']
collection = db['options_2']

sql = "select name from model_summary group by features order by mean_2 desc"
model_names = pd.read_sql(sql, conn)

for model_name in model_names:
    sql = 'select * from trades where name == "%s"' % model_name
    trades = pd.read_sql(sql, conn)
    trades = trades[trades['state']==2]
    print(trades)
    for _, trade in trades.iterrows():
        
        buy_date = trade['buy_date'].split(' ')[0].replace('-','')
        buy_price = float(trade['buy_price'])
        sell_price = float(trade['sell_price'])
        
        options = collection.find( { 
                                'Root': 'SPY', 
                                'Update_Date': buy_date,
                                'Open_Int': {'$gte': 1},
                                'Bid': {'$gt': 0},
                                'Ask': {'$gt': 0},
                                'Type': 'c'
                                })
        options = pd.DataFrame.from_dict(options)
        if options.empty:
            continue
        buy_option = options[options['iteration']==1]
        buy_option = buy_option[buy_option['Strike']<buy_price].sort_values(by='Strike')
        
        expirations = buy_option.sort_values(by=['Expiry'])['Expiry'].unique()
        
        expiration = expirations[0]

        buy_option = buy_option[buy_option['Expiry']==expiration].sort_values(by='Ask').head(1)
        
        option_id = buy_option['Symbol'].values[0]
        
        sell_option = options[options['Symbol']==option_id].tail(1)
        print(buy_price, sell_price)
        print(buy_option)
        print(sell_option)
        input()
        




"""
model_name = 'scummy-emerald-mandrill'

conn = sqlite3.connect('hidden_states.db')
client = MongoClient('192.168.1.142')
db = client['finance']
collection = db['options_2']
sql = 'select * from hidden_states_models_trades_technology where name == "%s"' % model_name
df = pd.read_sql(sql, conn)

expiry_number = 2

traded_options = []
for ticker, ticker_df in df.groupby(by=['ticker']):
    
    min_stock_price = ticker_df['buy_price'].min()
    max_stock_price = ticker_df['buy_price'].max()
    
    
    
    options = collection.find( { 
                                'Root': ticker, 
                                'Update_Date': {'$gte': '20190101'},
                                '$and': [ {'Strike': {'$gte': min_stock_price*.8}}, {'Strike': {'$lte': max_stock_price*1.2}} ],
                                '$or': [ {'iteration': {'$gte': 20}}, {'iteration': {'$lte': 8}} ],
                                'Open_Int': {'$gte': 10},
                                'Bid': {'$gte': .5},
                                'Ask': {'$gte': .5},
                                'Vol': {'$gte': 0},
                                'Type': 'c'
                                })
    options = pd.DataFrame.from_dict(options)
    print(options)
    options = options.sort_values(by=['Expiry'])
    

    for _, trade in ticker_df.iterrows():
        
        buy_date = trade['buy_date'].split(' ')[0].replace('-','')
        buy_price = float(trade['buy_price'])
        if trade['sell_date'] is None:
            sell_date = '20200226'
        else:
            sell_date = trade['sell_date'].split(' ')[0].replace('-','')
        sell_price = float(trade['sell_price'])
        
        # get all options for this buy date
        possible_options = options[ (options['Update_Date']==buy_date) ].sort_values(by='Strike')
        
        # select the option that has the nearest in the money strike price
        strike_price = possible_options[ possible_options['Strike'] < buy_price ]['Strike'].tail(1).values[0]

        possible_options = possible_options[possible_options['Strike'] == strike_price].sort_values(by='Expiry')
        
        # get expiration
        expirations = possible_options.sort_values(by=['Expiry'])['Expiry'].unique()
        
        expiration = expirations[expiry_number]

        possible_options = possible_options[possible_options['Expiry']==expiration].sort_values(by='iteration')
        
        selected_option = possible_options.head(2).tail(1)
        print(selected_option)
        symbol = selected_option['Symbol'].values[0]
        #print(symbol)
        ending_option = options[ (options['Symbol'] == symbol) & (options['Update_Date'] <= sell_date) ].sort_values(by=['Update_Date'])
        ending_option = ending_option.tail(1)
        
        option_start_price = selected_option['Ask'].values[0]
        option_end_price = ending_option['Bid'].values[0]
        
        option_percent_change = ( option_end_price - option_start_price) / option_start_price 
        stock_percent_change = ( sell_price - buy_price ) / buy_price

        traded_options.append( [ticker, buy_date, sell_date, stock_percent_change, option_percent_change] )
        traded_options_df = pd.DataFrame(traded_options, columns = ['ticker', 'buy_date', 'sell_date', 'stock_change', 'option_change'])
        print(traded_options_df)
        print(traded_options_df['option_change'].describe())
        input()


"""