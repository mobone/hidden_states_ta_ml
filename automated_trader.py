
from pipeline_hmm import pipeline
import alpaca_trade_api as tradeapi
from math import floor
import logging

logging.basicConfig(filename='/tmp/trader.log', level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(message)s')


"""
# TODO: use threading
tAMO = threading.Thread(target=self.awaitMarketOpen)
tAMO.start()
tAMO.join()
"""

class automated_trader():
    def __init__(self):
        self.model_name = "lumpy-linen-civet"
        
        self.short_symbol = 'QID'       # 2x short
        self.regular_symbol = 'QLD'     # 2x long
        self.strong_symbol = 'TQQQ'     # 3x long
        
        
        self.api = tradeapi.REST(
                                'PKY80T33QRG8ZMQPKI56',
                                'TDDNW7Je2mnmPQ7DRFR8CfbYHsniu5Qzk3we3LrQ',
                                'https://paper-api.alpaca.markets'
                                )


        self.held_shares = {}
        
        self.get_stock_data(self.short_symbol)
        self.get_stock_data(self.regular_symbol)
        self.get_stock_data(self.strong_symbol)

        self.get_todays_prediction()
        self.get_account_info()
        self.make_trades()

    def get_todays_prediction(self):
        x = pipeline(model_name = self.model_name)
        self.todays_prediction = x.new_predictions[['date', 'close', 'state']]
        logging.info('got todays state prediction')
        logging.info('\n'+str(self.todays_prediction['date','close','state'].tail(1)))

    def get_account_info(self):
        logging.info('getting account info from alpaca')
        self.account = self.api.get_account()
        self.account_equity = self.account.equity
        self.positions = self.api.list_positions()
        for position in self.positions:
            self.held_shares[position.symbol] = {'num_currently_held': position.qty}
        logging.info('got account info from alpaca')
        

    def make_trades(self):
        logging.info('starting to make trades')
        days = self.todays_prediction.tail(10)
        today = self.todays_prediction.tail(1)

        # check if we have a short position
        position = self.api.get_position(self.short_symbol)
        if position.qty != 0:
            self.short_qty = position.qty
            self.short_held = True
            continue

        if int(today['state'])==0:
            # sell everything 
            logging.info('state is bad, selling all holdings')
            for symbol, held_shares in self.held_shares.items():
                held_shares[symbol]['target_num_of_shares'] = 0
                difference = held_shares['target_num_of_shares'] - held_shares['num_currently_held']
                if difference>0:
                    logging.info('no need to sell shares for %s as the difference is %s' % ( symbol, difference ))
                    continue
                else:
                    logging.info('selling %s shares for %s.' % ( difference, symbol ))
                    submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'sell')
            
            if self.short_held == True:
                logging.info('already holding a short position. exiting')    
                return
            # place short trade
            get_target_num_shares(self.short_symbol, percent = 0.10)
            num_shares = self.held_shares[self.short_symbol]['target_num_of_shares']
            logging.info('placing short trade of %s shares for %s' % ( num_shares, self.short_symbol ))
            submit_order_wrapper(self.short_symbol, num_shares, num_shares, 'sell')
            return

        # close the short position 
        if self.short_held == True:
            submit_order_wrapper(self.short_symbol, self.short_qty, self.short_qty, 'buy')

        # find out the counts to determine the percent of account to use for each ETF
        num_regular = days[days['state']==1.0]['state'].count()
        num_strong = days[days['state']==2.0]['state'].count()

        # always hold one of the 2x or 3x ETF
        if num_strong == 0:
            num_strong = 1

        # convert counts to percents to be used against the bank balance
        regular_percent = num_regular / sum([num_regular, num_strong])
        strong_percent = num_strong / sum([num_regular, num_strong])

        if regular_percent>0:
            self.get_target_num_shares(self.regular_symbol, regular_percent)
        if strong_percent>0:
            self.get_target_num_shares(self.strong_symbol, strong_percent)
        
        # check if any of the shares need to be sold
        for symbol, held_shares in self.held_shares.items():
            difference = held_shares['target_num_of_shares'] - held_shares['num_currently_held']
            if difference<0:
                submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'sell')

        # check if any of the symbols need to be bought
        for symbol, held_shares in self.held_shares.items():
            difference = held_shares['target_num_of_shares'] - held_shares['num_currently_held']
            if difference>0:
                submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'buy')

        

    def get_target_num_shares(self, symbol, percent):
        # get amount of cash to use
        equity_to_use = self.account_equity * percent

        # get current share price
        symbol_bars = api.get_barset(symbol, 'minute', 1).df.iloc[0]
        current_price = symbol_bars[symbol]['close']
        self.current_prices[symbol] = current_price

        # get number of shares we should hold
        num_shares = floor( current_price * equity_to_use )

        self.held_shares[symbol]['target_num_of_shares'] = num_shares



    def submit_order_wrapper(self, symbol, num_shares, target_num_of_shares, side)
        current_price = self.current_prices[symbol]
        logging.info('submitting %s order for %s shares for %s. target number of shares is %s' % ( side, num_shares, symbol, target_num_of_shares ))
        order = api.submit_order( 
                                  symbol=symbol, 
                                  qty = abs(num_shares), 
                                  side = side, 
                                  type = 'limit', 
                                  time_in_force = 'gtc', 
                                  limit_price = round(current_price*.9, 2)
                                 )
        logging.info('order submitted')
        for i in range(10):
            # wait for order to clear
            sleep(1)

            position = self.api.get_position(symbol)
            if position.qty == target_num_of_shares:
                logging.info('order successfully filled')
                return
            else:
                logging.info('order has not been filled yet')
                sleep(2)
        
        logging.info('order failed to be filled')
        