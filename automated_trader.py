
from pipeline_hmm import pipeline
import alpaca_trade_api as tradeapi
from math import floor
import logging
from time import sleep
import pandas as pd
from threading import Thread

logging.basicConfig(filename='./trader.log', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s')


class automated_trader():
    def __init__(self):

        logging.info('started automated trader')
        self.model_name = "lumpy-linen-civet"
        
        self.use_margin = True

        self.short_symbol = 'QID'       # 2x short
        self.regular_symbol = 'QLD'     # 2x long
        self.strong_symbol = 'TQQQ'     # 3x long
        
        
        # nick
        self.api = tradeapi.REST(
                                'PK96YT85KHLEA2E5JYRK',
                                'e80TjkGBk4w6Nnch2mlCggGhWih7aYudDmHaDdQ0',
                                'https://paper-api.alpaca.markets'
                                )

        # tony
        
        self.api = tradeapi.REST(
                                'PKSRALP8NRAJ9AFBMN0O',
                                'Mu2Gv/V6AsWgWrGA48G6O2EjODh5DohpmklSQcZ6',
                                'https://paper-api.alpaca.markets'
                                )
        

        self.held_shares = {}
        self.current_prices = {}

        self.get_todays_prediction()
        self.get_account_info()
        self.make_trades()
        logging.info('all required trades submitted. automated trader exiting')
        logging.info('=============================================')


    def get_todays_prediction(self):
        x = pipeline(model_name = self.model_name)
        self.todays_prediction = x.new_predictions[['date', 'close', 'state']]
        logging.info('got todays state prediction')
        logging.info('\n'+str(
                                self.todays_prediction.tail(1)
                             ))
        logging.info('heres the last 10 days of predictions')
        logging.info('\n' + str(
                                 self.todays_prediction.tail(10)
                                ))


    def get_account_info(self):
        logging.info('getting account info from alpaca')
        
        self.account = self.api.get_account()
        self.account_equity = float(self.account.equity)
        #self.buying_power = float(self.account.buying_power)
        logging.info('got account value of %s' % self.account_equity)
        
        self.positions = self.api.list_positions()

        symbols = [self.short_symbol, self.regular_symbol, self.strong_symbol]
        for symbol in symbols:
            self.held_shares[symbol] = {'num_currently_held': 0, 'target_num_of_shares': 0}

        for position in self.positions:
            self.held_shares[position.symbol]['num_currently_held'] = position.qty
            #print('got %s shares currently held for %s' % (position.qty, position.symbol) )
            logging.info('got %s shares currently held for %s' % (position.qty, position.symbol) )


    def make_trades(self):
        logging.info('starting to make trades')
        self.days = self.todays_prediction.tail(10)
        self.today = self.todays_prediction.tail(1)

        # states used for testing
        #self.days['state']=1
        #self.today['state']=1

        self.get_current_prices()
        self.get_current_positions()
        self.get_equity_percents()

        self.get_target_num_shares(self.short_symbol, self.short_percent)
        self.get_target_num_shares(self.regular_symbol, self.regular_percent)
        self.get_target_num_shares(self.strong_symbol, self.strong_percent)

        logging.info('before sells, currently held and target share counts')
        logging.info(pd.DataFrame.from_dict(self.held_shares))
        #print(pd.DataFrame.from_dict(self.held_shares))

        # place sell orders first
        sell_orders = []
        for symbol, counts in self.held_shares.items():
            difference = int(counts['target_num_of_shares']) - int(counts['num_currently_held'])
            if difference<0:
                #self.submit_order_wrapper(symbol, difference, 'sell')
                sell_orders.append([symbol, difference, 'sell'])
        self.submit_order_threading(sell_orders)

        logging.info('after sells, currently held and target share counts')
        logging.info(pd.DataFrame.from_dict(self.held_shares))


        # then place buy orders
        buy_orders = []
        for symbol, counts in self.held_shares.items():
            difference = int(counts['target_num_of_shares']) - int(counts['num_currently_held'])
            if difference>0:
                #self.submit_order_wrapper(symbol, difference, 'buy')
                buy_orders.append([symbol, difference, 'buy'])
        self.submit_order_threading(buy_orders)


    def submit_order_threading(self, orders):
        threads = []
        for symbol, difference, side in orders:

            t = Thread(target=self.submit_order_wrapper, args = (symbol, difference, side, ))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def get_equity_percents(self):
        if float(self.today['state'])==0.0:
            self.sell_everything = True
        else:
            self.sell_everything = False
        # find out the counts to determine the percent of account to use for each ETF
        num_regular = self.days[self.days['state']==1.0]['state'].count()
        num_strong = self.days[self.days['state']==2.0]['state'].count()
        logging.info('got %s state counts for regular and %s state counts for strong' % ( num_regular, num_strong ) )

        # always hold one of the 2x or 3x ETF
        if num_strong == 0:
            num_strong = 1
            logging.info('bumping the strong count to always have a position')

        # convert counts to percents to be used against total account balance
        self.regular_percent = round(num_regular / sum([num_regular, num_strong]), 2)
        self.strong_percent = round(num_strong / sum([num_regular, num_strong]), 2)

        if self.sell_everything:
            self.regular_percent = 0
            self.strong_percent = 0
            self.short_percent = 0.25
        else:
            self.short_percent = 0

        
    def get_current_prices(self):
        symbols = [self.short_symbol, self.regular_symbol, self.strong_symbol]
        for symbol in symbols:
            # get current share price
            symbol_bars = self.api.get_barset(symbol, 'minute', 1).df.iloc[0]
            current_price = round(symbol_bars[symbol]['close'],2)
            self.current_prices[symbol] = current_price
            logging.info('got current price of $%s for %s' % ( current_price, symbol ))


    def get_current_positions(self):
        symbols = [self.short_symbol, self.regular_symbol, self.strong_symbol]
        for symbol in symbols:
            self.held_shares[symbol] = {'num_currently_held': 0, 'target_num_of_shares': 0}
            try:
                position = self.api.get_position(symbol)
                self.held_shares[symbol]['num_currently_held'] = position.qty
            except Exception as e:
                pass


    def get_target_num_shares(self, symbol, percent):
        print('starting equity', self.account_equity)
        # get amount of cash to use
        equity_to_use = round( float(self.account_equity) * percent,2)
        print('using percent', percent)
        logging.info('using $%s on %s' % (equity_to_use, symbol))

        # get current share price
        current_price = self.current_prices[symbol] * 1.05

        # if using margin, nearly double the equity to use
        if self.use_margin:
            equity_to_use = round(equity_to_use * 1.9,2)
        print('using equity', equity_to_use)
        # get number of shares we should hold
        num_shares = floor( equity_to_use / current_price )

        
        logging.info('determined we should hold %s shares of %s at $%s per share and using %s of equity' % (num_shares, symbol, current_price, percent*100) )

        self.held_shares[symbol]['target_num_of_shares'] = num_shares


    def submit_order_wrapper(self, symbol, num_shares, side):
        current_price = self.current_prices[symbol]
        num_shares = abs(int(num_shares))
        
        if side == 'buy':
            limit_price = round(current_price * 1.1,2)
        elif side == 'sell':
            limit_price = round(current_price * 0.90,2)
        
        print('submitting %s order for %s shares for %s at $%s per share for a total cost of $%s' % ( side, num_shares, symbol, limit_price, limit_price * num_shares ))
        logging.info('submitting %s order for %s shares for %s at $%s per share for a total cost of $%s' % ( side, num_shares, symbol, limit_price, limit_price * num_shares ))
        
        try:
            
            order = self.api.submit_order( 
                                    symbol=symbol, 
                                    qty = num_shares, 
                                    side = side, 
                                    type = 'limit', 
                                    time_in_force = 'day',
                                    extended_hours = True,
                                    limit_price = float(round(limit_price, 2))
                                    )
            logging.info('order submitted. got id of %s' % order.id)
            for _ in range(10):
                # wait for order to clear
                sleep(1)
                
                # check order status
                order = self.api.get_order(order.id)
                if order.status == 'filled':
                    #print('order %s filled' % order.id)
                    logging.info('order %s filled' % order.id)
                    return
                
            logging.info('order %s failed to be filled' % order.id)
        except Exception as e:
            #print(e)
            logging.error('got exception when submitting order\n%s' % e)

automated_trader()