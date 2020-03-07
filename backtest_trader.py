from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG
import pandas as pd


def Alert(values):
    #print(values)
    return values

class MyStrat(Strategy):

    def init(self):
        
        self.signal = self.I(Alert, self.data['State'])
        

    def next(self):
        
        today = self.signal[-1:][0]
        todays_price = self.data.Close
        #if today != 0 and self.position.is_short:
        #    self.position.close()
        if today == 0 and self.position.is_long:            
            self.position.close()

        if ( today > 0 ) and self.position.is_long == False:
            self.buy()
        
        
            #self.sell()
