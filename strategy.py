from __future__ import print_function

from pyalgotrade import strategy
from pyalgotrade.barfeed import quandlfeed, yahoofeed, csvfeed,googlefeed

from pyalgotrade.technical import ma
import pandas as pd

from pyalgotrade import dataseries
from pyalgotrade import technical
#from pyalgotrade.order import BUY, SELL
from math import floor
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.stratanalyzer import drawdown
from pyalgotrade.stratanalyzer import trades
from pyalgotrade.bar import Frequency
from pyalgotrade import plotter
import matplotlib

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument_1, instrument_2, states_instrument, smaPeriod=1):
        super(MyStrategy, self).__init__(feed, 20000)
        self.__position = None
        self.__instrument_1 = instrument_1
        self.__instrument_2 = instrument_2
        self.states_instrument = states_instrument
        self.last_state = None
        
        
        self.__my_indicator = ma.SMA(feed[states_instrument].getCloseDataSeries(), smaPeriod)

        self.usage = {}
        
    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info("BUY at $%.2f" % (execInfo.getPrice()))

    def onEnterCanceled(self, position):
        self.__position = None

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info("SELL at $%.2f" % (execInfo.getPrice()))
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()

    def onBars(self, bars):
        
        state = self.__my_indicator[-1]
        if state is None:
            return


        if state == self.last_state:
            return
        
        #print('===========')
        #print(state)
        
        #for i in [self.__instrument_1, self.__instrument_2]:
        #    print(i, self.getBroker().getShares(i))

        if state == 1:
            instrument = self.__instrument_1
            self.usage[self.__instrument_1] = .8
            self.usage[self.__instrument_2] = .2
            

        elif state == 2:
            instrument = self.__instrument_2
            self.usage[self.__instrument_1] = .2
            self.usage[self.__instrument_2] = .8
        
        #elif state == 3:
        #    self.usage[self.__instrument_1] = .2
        #    self.usage[self.__instrument_2] = .8
        
        elif state == 0:
            instrument = None
            
            
        if state==0:
            for instrument in [self.__instrument_1, self.__instrument_2]:
                bar = bars.getBar(instrument)
                close = bar.getClose()
                currentPos = self.getBroker().getShares(instrument) * -1
                if abs(currentPos)>0:
                    #print('selling', instrument, close * 0.9, currentPos)
                    self.limitOrder(instrument, close * 0.9, currentPos)
            #print(self.getBroker().getPositions())
            #input()
            self.last_state = state
            return


        for instrument in [self.__instrument_1, self.__instrument_2]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*0.9) )

            currentPos = self.getBroker().getShares(instrument)

            num_shares = num_shares - currentPos

            if num_shares<0:
                
                self.limitOrder(instrument, close * 0.9, num_shares)
                

                #print('limit sell order', self.getBroker().getEquity(), usage, instrument, close * 0.9, num_shares, close *0.9 * num_shares)
                #print(self.getBroker().getCash())
        
        for instrument in [self.__instrument_1, self.__instrument_2]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*1.1) )
            
            currentPos = self.getBroker().getShares(instrument)
            
            num_shares = num_shares - currentPos
            
            if num_shares>0:
                self.limitOrder(instrument, close * 1.1, num_shares)
                
                #print('limit buy order', self.getBroker().getEquity(), usage, instrument, close * 1.1, num_shares, close * 1.1 * num_shares)

        #print(self.getBroker().getPositions())
        #input()
        """

        bar = bars.getBar(instrument)
        close = bar.getClose()
        

        num_shares = floor( self.getBroker().getCash() / (close*1.1) )
        
        currentPos = self.getBroker().getShares(instrument)
        
        num_shares = num_shares - currentPos
        
        if num_shares>0:
            self.limitOrder(instrument, close * 1.1, num_shares)
        """
        #print(state)
        #for i in [self.__instrument_1, self.__instrument_2]:
        #    print(i, self.getBroker().getShares(i))
        
        
        
        
        
        self.last_state = state
        """
        # If a position was not opened, check if we should enter a long position.
        if self.__position is None:
            if bar.getPrice() > self.__sma[-1]:
                # Enter a buy market order for 10 shares. The order is good till canceled.
                self.__position = self.enterLong(self.__instrument, 10, True)
        # Check if we have to exit the position.
        elif bar.getPrice() < self.__sma[-1] and not self.__position.exitActive():
            self.__position.exitMarket()
        """


def setup_strategy(files, name, smaPeriod=1):
    #from pyalgotrade.feed import csvfeed, yahoofeed

    # Load the bar feed from the CSV file
    #feed = csvfeed.GenericBarFeed(frequency=Frequency.DAY)
    feed = yahoofeed.Feed(Frequency.DAY)
    #feed = csvfeed.Feed("Date", "%Y-%m-%d")

    for sym, filename in files:
        #print('loading', sym, filename)
        feed.addBarsFromCSV(sym, filename)
        #print(list( feed[sym].getAdjCloseDataSeries() ) )
    
    
    instrument_1 = files[0][0]
    instrument_2 = files[1][0]
    states_instrument = files[2][0]
    
    
    # Evaluate the strategy with the feed.
    myStrategy = MyStrategy(feed, instrument_1, instrument_2, states_instrument, smaPeriod)
    from pyalgotrade.stratanalyzer import returns
    # Attach different analyzers to a strategy before executing it.
    retAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(retAnalyzer)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    myStrategy.attachAnalyzer(sharpeRatioAnalyzer)
    drawDownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawDownAnalyzer)
    tradesAnalyzer = trades.Trades()
    myStrategy.attachAnalyzer(tradesAnalyzer)

    # Attach the plotter to the strategy.
    plt = plotter.StrategyPlotter(myStrategy)
    # Include the SMA in the instrument's subplot to get it displayed along with the closing prices.
    #plt.getInstrumentSubplot("orcl").addDataSeries("SMA", myStrategy.getSMA())
    # Plot the simple returns on each bar.
    plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", retAnalyzer.getReturns())
    

    #plt.plot()
    #

    # Run the strategy.
    myStrategy.run()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savePlot('./plots/%s.png' % ('backtest_'+name))

    del plt
        
    results = {}
    results['final_value'] = myStrategy.getResult()
    results['cum_returns'] = retAnalyzer.getCumulativeReturns()[-1] * 100
    results['sharpe_ratio'] = sharpeRatioAnalyzer.getSharpeRatio(0.05)
    results['max_drawdown_%'] = drawDownAnalyzer.getMaxDrawDown() * 100
    results['longest_drawdown'] = str(drawDownAnalyzer.getLongestDrawDownDuration())

    results['total_trades'] = tradesAnalyzer.getCount()
    results['profitable_trades'] = tradesAnalyzer.getProfitableCount()
    results['win_rate'] = tradesAnalyzer.getProfitableCount() / tradesAnalyzer.getCount()

    profits = tradesAnalyzer.getAll()
    results['avg_profit_$'] = profits.mean()
    results['std_profit_$'] = profits.std()
    results['max_profit_$'] = profits.max()
    results['min_profit_$'] = profits.min()
    
    returns = tradesAnalyzer.getAllReturns()
    results['avg_profit_%'] = returns.mean() * 100
    results['std_profit_%'] = returns.std() * 100
    results['max_profit_%'] = returns.max() * 100
    results['min_profit_%'] = returns.min() * 100
    results = pd.DataFrame.from_dict(results, orient='index')
    #print(results)

    return results

if __name__ == "__main__":
    #run_strategy(1)
    smaPeriod = 1

    # Load the bar feed from the CSV file
    feed = yahoofeed.Feed(Frequency.DAY)
    """
    feed.addBarsFromCSV("QLD", "QLD_2.csv")
    feed.addBarsFromCSV("TQQQ", "TQQQ_2.csv")
    feed.addBarsFromCSV("QLD_state", "QLD_2_with_states.csv")
    """
    feed.addBarsFromCSV("QLD", "QLD.csv")
    feed.addBarsFromCSV("TQQQ", "TQQQ.csv")
    feed.addBarsFromCSV("QLD_state", "QLD_with_states.csv")

    
    # Evaluate the strategy with the feed.
    myStrategy = MyStrategy(feed, "QLD", 'TQQQ', 'QLD_state')

    # Attach different analyzers to a strategy before executing it.
    retAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(retAnalyzer)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    myStrategy.attachAnalyzer(sharpeRatioAnalyzer)
    drawDownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawDownAnalyzer)
    tradesAnalyzer = trades.Trades()
    myStrategy.attachAnalyzer(tradesAnalyzer)

    # Run the strategy.
    myStrategy.run()

    print("Final portfolio value: $%.2f" % myStrategy.getResult())
    print("Cumulative returns: %.2f %%" % (retAnalyzer.getCumulativeReturns()[-1] * 100))
    print("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.05)))
    print("Max. drawdown: %.2f %%" % (drawDownAnalyzer.getMaxDrawDown() * 100))
    print("Longest drawdown duration: %s" % (drawDownAnalyzer.getLongestDrawDownDuration()))

    print("")
    print("Total trades: %d" % (tradesAnalyzer.getCount()))
    if tradesAnalyzer.getCount() > 0:
        profits = tradesAnalyzer.getAll()
        print("Avg. profit: $%2.f" % (profits.mean()))
        print("Profits std. dev.: $%2.f" % (profits.std()))
        print("Max. profit: $%2.f" % (profits.max()))
        print("Min. profit: $%2.f" % (profits.min()))
        returns = tradesAnalyzer.getAllReturns()
        print("Avg. return: %2.f %%" % (returns.mean() * 100))
        print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
        print("Max. return: %2.f %%" % (returns.max() * 100))
        print("Min. return: %2.f %%" % (returns.min() * 100))

    print("")
    print("Profitable trades: %d" % (tradesAnalyzer.getProfitableCount()))
    if tradesAnalyzer.getProfitableCount() > 0:
        profits = tradesAnalyzer.getProfits()
        print("Avg. profit: $%2.f" % (profits.mean()))
        print("Profits std. dev.: $%2.f" % (profits.std()))
        print("Max. profit: $%2.f" % (profits.max()))
        print("Min. profit: $%2.f" % (profits.min()))
        returns = tradesAnalyzer.getPositiveReturns()
        print("Avg. return: %2.f %%" % (returns.mean() * 100))
        print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
        print("Max. return: %2.f %%" % (returns.max() * 100))
        print("Min. return: %2.f %%" % (returns.min() * 100))

    print("")
    print("Unprofitable trades: %d" % (tradesAnalyzer.getUnprofitableCount()))
    if tradesAnalyzer.getUnprofitableCount() > 0:
        losses = tradesAnalyzer.getLosses()
        print("Avg. loss: $%2.f" % (losses.mean()))
        print("Losses std. dev.: $%2.f" % (losses.std()))
        print("Max. loss: $%2.f" % (losses.min()))
        print("Min. loss: $%2.f" % (losses.max()))
        returns = tradesAnalyzer.getNegativeReturns()
        print("Avg. return: %2.f %%" % (returns.mean() * 100))
        print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
        print("Max. return: %2.f %%" % (returns.max() * 100))
        print("Min. return: %2.f %%" % (returns.min() * 100))

