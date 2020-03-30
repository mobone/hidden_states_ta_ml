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
import numpy as np
import yfinance

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument_1, instrument_2, short_instrument, states_instrument, with_short=False, smaPeriod=1):
        super(MyStrategy, self).__init__(feed, 20000)
        self.__position = None
        self.__instrument_1 = instrument_1
        self.__instrument_2 = instrument_2
        self.__short_instrument = short_instrument

        self.with_short = with_short

        self.states_instrument = states_instrument
        
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

        if state == 1:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 1
            self.usage[self.__short_instrument] = 0 
            

        elif state == 2:
            
            self.usage[self.__instrument_1] = 1
            self.usage[self.__instrument_2] = 0
            self.usage[self.__short_instrument] = 0 
        
        elif state == 3:
            self.usage[self.__instrument_1] = .5
            self.usage[self.__instrument_2] = 0
            self.usage[self.__short_instrument] = 0 
        
        elif state == 0:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 0
            if self.with_short:
                self.usage[self.__short_instrument] = .5
            else:
                self.usage[self.__short_instrument] = 0

        for instrument in [self.__instrument_1, self.__instrument_2, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*.9) )

            currentPos = self.getBroker().getShares(instrument)

            num_shares = int(num_shares - currentPos)

            if num_shares<0:
                
                #self.limitOrder(instrument, close * 0.9, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)
        
        for instrument in [self.__instrument_1, self.__instrument_2, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*1.1) )
            
            currentPos = self.getBroker().getShares(instrument)
            
            num_shares = int(num_shares - currentPos)
            
            if num_shares>0:
                #self.limitOrder(instrument, close * 1.1, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)

class MyStrategy_2(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument_1, instrument_2, short_instrument, states_instrument, with_short=False, smaPeriod=1):
        super(MyStrategy_2, self).__init__(feed, 20000)
        self.__position = None
        self.__instrument_1 = instrument_1
        self.__instrument_2 = instrument_2
        self.__short_instrument = short_instrument

        self.with_short = with_short

        self.states_instrument = states_instrument
        
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

        if state == 1:
            self.usage[self.__instrument_1] = .5
            self.usage[self.__instrument_2] = 0
            self.usage[self.__short_instrument] = 0
            

        elif state == 2:
            
            self.usage[self.__instrument_1] = 1
            self.usage[self.__instrument_2] = 0
            self.usage[self.__short_instrument] = 0 
        
        elif state == 3:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 1
            self.usage[self.__short_instrument] = 0
        
        elif state == 0:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 0
            if self.with_short:
                self.usage[self.__short_instrument] = .5
            else:
                self.usage[self.__short_instrument] = 0

        for instrument in [self.__instrument_1, self.__instrument_2, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*.9) )

            currentPos = self.getBroker().getShares(instrument)

            num_shares = int(num_shares - currentPos)

            if num_shares<0:
                
                #self.limitOrder(instrument, close * 0.9, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)
        
        for instrument in [self.__instrument_1, self.__instrument_2, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*1.1) )
            
            currentPos = self.getBroker().getShares(instrument)
            
            num_shares = int(num_shares - currentPos)
            
            if num_shares>0:
                #self.limitOrder(instrument, close * 1.1, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)



class MyStrategy_3(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument_1, instrument_2, short_instrument, states_instrument, with_short=False, smaPeriod=1):
        super(MyStrategy_3, self).__init__(feed, 20000)
        self.__position = None
        self.__instrument_1 = instrument_1
        self.__instrument_2 = instrument_2
        self.__short_instrument = short_instrument

        self.with_short = with_short

        self.states_instrument = states_instrument
        
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

        if state == 1:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = .5
            self.usage[self.__short_instrument] = 0
            

        elif state == 2:
            
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 1
            self.usage[self.__short_instrument] = 0 
        
        elif state == 3:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 1
            self.usage[self.__short_instrument] = 0 
        
        elif state == 0:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 0
            if self.with_short:
                self.usage[self.__short_instrument] = .5
            else:
                self.usage[self.__short_instrument] = 0

        for instrument in [self.__instrument_1, self.__instrument_2, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*.9) )

            currentPos = self.getBroker().getShares(instrument)

            num_shares = int(num_shares - currentPos)

            if num_shares<0:
                
                #self.limitOrder(instrument, close * 0.9, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)
        
        for instrument in [self.__instrument_1, self.__instrument_2, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*1.1) )
            
            currentPos = self.getBroker().getShares(instrument)
            
            num_shares = int(num_shares - currentPos)
            
            if num_shares>0:
                #self.limitOrder(instrument, close * 1.1, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)




class AccuracyStrat(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument_1, short_instrument, states_instrument, svc_states_instrument, with_short=False, smaPeriod=1):
        super(AccuracyStrat, self).__init__(feed, 20000)
        self.__position = None
        self.__instrument_1 = instrument_1
        
        self.__short_instrument = short_instrument
        

        self.with_short = with_short

        self.states_instrument = states_instrument
        self.svc_states_instrument = svc_states_instrument
        
        self.__my_indicator = ma.SMA(feed[states_instrument].getCloseDataSeries(), smaPeriod)
        self.__my_svc_indicator = ma.SMA(feed[svc_states_instrument].getCloseDataSeries(), smaPeriod)

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
        svc_state = self.__my_svc_indicator[-1]
        if state is None:
            return

        self.usage[self.__instrument_1] = 0
        self.usage[self.__short_instrument] = 0

        if svc_state == 1 and state >= 1:
            self.usage[self.__instrument_1] = 1
            self.usage[self.__short_instrument] = 0
            

        elif svc_state == 0:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__short_instrument] = 0

        elif svc_state == -1 and state == 0:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__short_instrument] = 1


        for instrument in [self.__instrument_1, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*.9) )

            currentPos = self.getBroker().getShares(instrument)

            num_shares = int(num_shares - currentPos)

            if num_shares<0:
                
                #self.limitOrder(instrument, close * 0.9, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)
        
        for instrument in [self.__instrument_1, self.__short_instrument]:
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*1.1) )
            
            currentPos = self.getBroker().getShares(instrument)
            
            num_shares = int(num_shares - currentPos)
            
            if num_shares>0:
                #self.limitOrder(instrument, close * 1.1, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)


def setup_strategy(files, name, strategy, with_short = False, smaPeriod=1):
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
    short_instrument = files[1][0]
    states_instrument = files[2][0]
    svc_states_instrument = files[3][0]
    
    print('got these instruments', instrument_1, short_instrument, states_instrument, svc_states_instrument)
    print(files)
    # Evaluate the strategy with the feed.
    myStrategy = strategy(feed, instrument_1, short_instrument, states_instrument, svc_states_instrument, with_short = with_short, smaPeriod = smaPeriod)
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
    plt.savePlot('./all_models_plots/%s.png' % ('backtest_'+name))

    del plt
        
    results = {}
    results['final_value'] = myStrategy.getResult()
    results['cum_returns'] = retAnalyzer.getCumulativeReturns()[-1] * 100
    results['sharpe_ratio'] = sharpeRatioAnalyzer.getSharpeRatio(0.05)
    results['max_drawdown_%'] = drawDownAnalyzer.getMaxDrawDown() * 100
    results['longest_drawdown'] = str(drawDownAnalyzer.getLongestDrawDownDuration())

    results['total_trades'] = tradesAnalyzer.getCount()
    results['profitable_trades'] = tradesAnalyzer.getProfitableCount()

    try:
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
    except Exception as e:
        print('backtest exception', e)
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
    #feed.addBarsFromCSV("QLD", "QLD.csv")
    #feed.addBarsFromCSV("TQQQ", "TQQQ.csv")
    #feed.addBarsFromCSV("QLD_state", "QLD_with_states.csv")

    histories = []
    for ticker in ['QLD', 'TQQQ', 'TQQQ_with_states']:
        history = yfinance.Ticker(ticker.replace('_with_states', '')).history(period='8y', auto_adjust=False)
        #print(history)
        histories.append( [ticker, history] )
        
    states = list(np.random.randint(0,2, size=len(histories[2][1])))
    
    histories[2][1]['Close'] = states
    histories[2][1]['Low'] = states


    for ticker, history in histories:
        history.to_csv(ticker+'.csv')
        print('adding', ticker)
        feed.addBarsFromCSV(ticker, ticker+'.csv')
    
        
    
    # Evaluate the strategy with the feed.
    myStrategy = MyStrategy(feed, "QLD", 'TQQQ', 'TQQQ_with_states')

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

