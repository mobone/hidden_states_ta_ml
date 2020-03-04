
from pipeline_hmm import pipeline
import alpaca_trade_api as tradeapi
from math import floor
import logging
from time import sleep
import pandas as pd
logging.basicConfig(filename='/tmp/trader.log', level=logging.INFO)
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
                                'PKMOAVSZFGD24YGBSEDE',
                                'DzOHQ6DjB1zh9SsZvjKGiRdUB0gFWzzNuYnPROzF',
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
        self.account_equity = self.account.equity
        logging.info('got account balance of %s' % self.account_equity)

        self.positions = self.api.list_positions()

        symbols = [self.short_symbol, self.regular_symbol, self.strong_symbol]
        for symbol in symbols:
            self.held_shares[symbol] = {'num_currently_held': 0, 'target_num_of_shares': 0}

        for position in self.positions:
            self.held_shares[position.symbol]['num_currently_held'] = position.qty
            #print('got %s shares currently held for %s' % (position.qty, position.symbol) )
            #logging.info('got %s shares currently held for %s' % (position.qty, position.symbol) )

        #print('currently held', self.held_shares)

        
        
        
        

    def make_trades(self):
        logging.info('starting to make trades')
        self.days = self.todays_prediction.tail(10)
        self.today = self.todays_prediction.tail(1)

        # test states
        self.days['state']=1
        self.today['state']=2

        self.get_current_prices()
        self.get_current_positions()
        self.get_equity_percents()

        self.get_target_num_shares(self.short_symbol, self.short_percent)
        self.get_target_num_shares(self.regular_symbol, self.regular_percent)
        self.get_target_num_shares(self.strong_symbol, self.strong_percent)

        print(pd.DataFrame.from_dict(self.held_shares))

        # place sell orders first
        for symbol, counts in self.held_shares.items():
            difference = int(counts['target_num_of_shares']) - int(counts['num_currently_held'])
            if difference<0:
                self.submit_order_wrapper(symbol, difference, 'sell')

        # then place buy orders
        for symbol, counts in self.held_shares.items():
            difference = int(counts['target_num_of_shares']) - int(counts['num_currently_held'])
            if difference>0:
                self.submit_order_wrapper(symbol, difference, 'buy')




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
        self.regular_percent = num_regular / sum([num_regular, num_strong])
        self.strong_percent = num_strong / sum([num_regular, num_strong])

        if self.sell_everything:
            self.regular_percent = 0
            self.strong_percent = 0
            self.short_percent = 0.25
        else:
            self.short_percent = 0
        

        """
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

        # close the short position as the current state is not 0
        if self.short_held == True and int(today['state']!=0):
            logging.info('closing short position of %s shares' % self.short_qty)
            self.submit_order_wrapper(self.short_symbol, self.short_qty, self.short_qty, 'sell')

        if int(today['state'])==0:
            # sell everything 
            logging.info('state is bad, selling all holdings')
            
            for symbol, held_shares in self.held_shares.items():
                held_shares['target_num_of_shares'] = 0
                difference = int(held_shares['target_num_of_shares']) - int(held_shares['num_currently_held'])
                if difference>=0:
                    print('no need to sell shares')
                    logging.info('no need to sell shares for %s as the difference is %s' % ( symbol, difference ))
                    continue
                else:
                    
                    logging.info('selling %s shares for %s.' % ( difference, symbol ))
                    self.submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'sell')
            
            if self.short_held == True:
                print('aready held short')
                logging.info('already holding a short position. exiting')    
                return
            # place short trade
            self.get_target_num_shares(self.short_symbol, percent = 0.20)
            num_shares = self.held_shares[self.short_symbol]['target_num_of_shares']
            print('buying short shares')
            logging.info('placing short trade of %s shares for %s' % ( num_shares, self.short_symbol ))
            self.submit_order_wrapper(self.short_symbol, num_shares, num_shares, 'buy')
            return

        

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
            print(self.held_shares)
            print(symbol)
            print(held_shares)
            difference = int(held_shares['target_num_of_shares']) - int(held_shares['num_currently_held'])
            logging.info('determined we need to sell %s shares of %s for a target of %s shares' % ( abs(difference), symbol, held_shares['target_num_of_shares'] ) )
            if difference<0:
                self.submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'sell')

        # check if any of the symbols need to be bought
        for symbol, held_shares in self.held_shares.items():
            difference = int(held_shares['target_num_of_shares']) - int(held_shares['num_currently_held'])
            logging.info('determined we need to buy %s shares of %s for a target of %s shares' % ( abs(difference), symbol, held_shares['target_num_of_shares'] ) )
            if difference>0:
                self.submit_order_wrapper(symbol, difference, held_shares['target_num_of_shares'], 'buy')
        """
        
    def get_current_prices(self):
        symbols = [self.short_symbol, self.regular_symbol, self.strong_symbol]
        for symbol in symbols:
            # get current share price
            symbol_bars = self.api.get_barset(symbol, 'minute', 1).df.iloc[0]
            current_price = symbol_bars[symbol]['close']
            self.current_prices[symbol] = current_price
        
    def get_current_positions(self):
        symbols = [self.short_symbol, self.regular_symbol, self.strong_symbol]
        for symbol in symbols:
            self.held_shares[symbol] = {'num_currently_held': 0, 'target_num_of_shares': 0}
            try:
                position = self.api.get_position(symbol)
                self.held_shares[symbol]['num_currently_held'] = position.qty
                #print('found position of %s shares for %s' % ( position.qty, symbol ) )
            except Exception as e:
                print("no position exists for", symbol)
                print(e)
                pass



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

        self.held_shares[symbol]['target_num_of_shares'] = num_shares



    def submit_order_wrapper(self, symbol, num_shares, side):
        current_price = self.current_prices[symbol]
        num_shares = abs(int(num_shares))
        print('submitting %s order for %s shares for %s' % ( side, num_shares, symbol ))
        #logging.info('submitting %s order for %s shares for %s. target number of shares is %s' % ( side, num_shares, symbol ))
        if side == 'buy':
            limit_price = current_price * 1.15
        elif side == 'sell':
            limit_price = current_price * 0.85
        
        
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
            logging.info('order submitted')
            for i in range(10):
                # wait for order to clear
                sleep(1)
                
                order = self.api.get_order(order.id)
                if order.status == 'filled':
                    print('order %s filled' % order.id)
                    return
                
            input()
                
            logging.info('order failed to be filled')
        except Exception as e:
            print(e)
            logging.error('got exception when submitting order\n%s' % e)

automated_trader()