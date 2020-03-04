
from pipeline_hmm import pipeline
import alpaca_trade_api as tradeapi
from math import floor
import logging
from time import sleep
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

        logging.info('started automated trader')
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

    def get_account_info(self):
        logging.info('getting account info from alpaca')
        
        self.account = self.api.get_account()
        self.account_equity = self.account.equity
        logging.info('got account balance of %s' % self.account_equity)

        self.positions = self.api.list_positions()
        for position in self.positions:
            self.held_shares[position.symbol] = {'num_currently_held': position.qty}
            logging.info('got %s shares currently held for %s' % (position.qty, position.symbol) )
        
        

    def make_trades(self):
        logging.info('starting to make trades')
        days = self.todays_prediction.tail(10)
        today = self.todays_prediction.tail(1)

        # check if we have a short position
        try:
            position = self.api.get_position(self.short_symbol)
            if position.qty != 0:
                self.short_qty = position.qty
                self.short_held = True
                logging.info('short position of %s shares exists for %s' % ( position.qty, self.short_symbol ) )
        except:
            logging.info('short position does not exist')
            self.short_held = False
            self.short_qty = 0

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
                    self.submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'sell')
            
            if self.short_held == True:
                logging.info('already holding a short position. exiting')    
                return
            # place short trade
            self.get_target_num_shares(self.short_symbol, percent = 0.20)
            num_shares = self.held_shares[self.short_symbol]['target_num_of_shares']
            logging.info('placing short trade of %s shares for %s' % ( num_shares, self.short_symbol ))
            self.submit_order_wrapper(self.short_symbol, num_shares, num_shares, 'buy')
            return

        # close the short position as the current state is not 0
        if self.short_held == True:
            logging.info('closing short position of %s shares' % self.short_qty)
            self.submit_order_wrapper(self.short_symbol, self.short_qty, self.short_qty, 'sell')

        # find out the counts to determine the percent of account to use for each ETF
        num_regular = days[days['state']==1.0]['state'].count()
        num_strong = days[days['state']==2.0]['state'].count()
        logging.info('got %s state counts for regular and %s state counts for strong' % ( num_regular, num_strong ) )

        # always hold one of the 2x or 3x ETF
        if num_strong == 0:
            num_strong = 1
            logging.info('bumping the strong count to always have a position')

        # convert counts to percents to be used against total account balance
        regular_percent = num_regular / sum([num_regular, num_strong])
        strong_percent = num_strong / sum([num_regular, num_strong])

        logging.info('using %s percent and %s percent of account for regular and strong' % ( round(regular_percent*100,2), round(strong_percent*100,2) ))

        if regular_percent>0:
            self.get_target_num_shares(self.regular_symbol, regular_percent)
        if strong_percent>0:
            self.get_target_num_shares(self.strong_symbol, strong_percent)
        
        # check if any of the shares need to be sold
        for symbol, held_shares in self.held_shares.items():
            difference = held_shares['target_num_of_shares'] - held_shares['num_currently_held']
            logging.info('determined we need to sell %s shares of %s for a target of %s shares' % ( abs(difference), symbol, held_shares['target_num_of_shares'] ) )
            if difference<0:
                self.submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'sell')

        # check if any of the symbols need to be bought
        for symbol, held_shares in self.held_shares.items():
            difference = held_shares['target_num_of_shares'] - held_shares['num_currently_held']
            logging.info('determined we need to buy %s shares of %s for a target of %s shares' % ( abs(difference), symbol, held_shares['target_num_of_shares'] ) )
            if difference>0:
                self.submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'buy')

        

        

    def get_target_num_shares(self, symbol, percent):
        # get amount of cash to use
        equity_to_use = round( float(self.account_equity) * percent,2)
        logging.info('using $%s on %s' % (equity_to_use, symbol))

        # get current share price
        symbol_bars = self.api.get_barset(symbol, 'minute', 1).df.iloc[0]
        current_price = symbol_bars[symbol]['close']
        self.current_prices[symbol] = current_price

        logging.info('got current share price of %s for %s' % (current_price, symbol) )

        # get number of shares we should hold
        num_shares = floor( equity_to_use / current_price )

        logging.info('determined we should hold %s shares of %s' % (num_shares, symbol) )

        self.held_shares[symbol] = {'target_num_of_shares': num_shares}



    def submit_order_wrapper(self, symbol, num_shares, target_num_of_shares, side):
        current_price = self.current_prices[symbol]
        logging.info('submitting %s order for %s shares for %s. target number of shares is %s' % ( side, num_shares, symbol, target_num_of_shares ))
        if side == 'buy':
            limit_price = current_price * 1.15
        elif side == 'sell':
            limit_price = current_price * 0.85

        try:
            order = self.api.submit_order( 
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
                try:
                    position = self.api.get_position(symbol)
                    if int(position.qty) == int(target_num_of_shares):
                        logging.info('order successfully filled')
                        return
                except Exception as e:
                    logging.error('got exception\n%e' % e)
                
                
                    logging.info('order has not been filled yet')
                    sleep(2)
            
            logging.info('order failed to be filled')
        except Exception as e:
            logging.error('got exception\n%s' % e)

automated_trader()