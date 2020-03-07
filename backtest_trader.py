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
        
        #if today != 0 and self.position.is_short:
        #    self.position.close()
        if today == 0 and self.position.is_long:            
            self.position.close()

        if ( today > 0 ) and self.position.is_long == False:
            self.buy()
        
        
            #self.sell()
        
        
class MyStratWithShort(Strategy):

    def init(self):
        
        self.signal = self.I(Alert, self.data['State'])
        

    def next(self):
        
        today = self.signal[-1:][0]
        
        # Do Sells first
        if today != 0 and self.position.is_short:
            self.position.close()

        if today == 0 and self.position.is_long:            
            self.position.close()

        # Then do buys

        if ( today > 0 ) and self.position.is_long == False:
            self.buy()
        
        if today == 0 and self.position.is_short == False:
            self.sell()
            #self.sell()
        
            
    
"""        
history = pd.read_csv('trades.csv')

history.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'State', 'State_Name']
history['Date'] = pd.to_datetime(history['Date'])
history = history.set_index('Date')

for col in history.columns:
    history[col] = pd.to_numeric(history[col], errors='ignore')




bt = Backtest(history, SmaCross, margin=1/2, cash=10000, commission=.00, trade_on_close=1)

output = bt.run()
print(output)
#bt.plot()
"""