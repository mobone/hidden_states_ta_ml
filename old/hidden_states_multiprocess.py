import pandas as pd
import yfinance
from sklearn import mixture as mix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import YearLocator, MonthLocator
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend
from ta_indicators import get_ta
import warnings
import time
import namegenerator
from multiprocessing import Process, Queue, Pool, cpu_count
import sqlite3
from time import sleep
import psutil
warnings.simplefilter("ignore")

# TODO: make plots; fix bug with dates?

class stock_predictor():
    def __init__(self, params): 
    
        
        
        self.ticker = params.get('ticker', 'QLD')
        self.name = params.get('name', namegenerator.gen())
        self.hold_length = params.get('hold_length', 1)
        self.period = params.get('period', '3y')
        self.k_feature_range = params.get('k_feature_range',range(4,10))
        self.num_features = params.get('num_features', 4)
        self.max_depth = params.get('max_depth', 5)
        self.scoring = params.get('scoring', 'neg_mean_squared_error')
        self.pattern = params.get('pattern', True)
        self.trade_other_states = params.get('trade_other_states', True)
        self.with_decision_tree = params.get('with_decision_tree', True)
        self.with_original = params.get('with_original', True)
        self.features = params.get('features', None)
        self.all_historic_data = pd.DataFrame()

        self.conn = sqlite3.connect('hidden_states.db')

        
        
        self.get_data()
        self.get_train_test()
        if self.features is None:  # generates features and models
            
            if self.with_decision_tree:
                self.run_sfs()
                if self.with_original:
                    
                    self.features = list(set( self.features + ['return', 'range', 'close'] ))
            else:
                self.features = ['return', 'range', 'close']
                self.num_features = 3
            #for subset in self.features_dict.keys():
            #    print(self.features_dict[subset])
            #    input()
            self.predict()
            self.get_results()
            self.store_results()
        elif self.features is not None:
            self.features = eval(self.features)
            #self.find_best_features()
            #if self.with_original:
            #        self.features = list(set( self.features + ['return', 'range', 'close'] ))
            self.predict()
            self.get_results()


        
    def get_data(self, with_decision_tree=False):
        ticker_data = yfinance.Ticker(self.ticker)
        ticker_data = ticker_data.history(period=self.period, auto_adjust=False)
        
        ticker_data.columns = map(str.lower, ticker_data.columns)
        
        if self.with_decision_tree==True:
            ticker_data = get_ta(ticker_data, pattern = self.pattern)
            ticker_data['return'] = ticker_data['close'].shift(-self.hold_length) / ticker_data['close'] - 1
        elif self.with_decision_tree==False:
            # TODO: test with this pct change instead of intraday change ?
            #ticker_data["return"] = ticker_data["close"].pct_change()
            ticker_data["return"] = ticker_data["close"] / ticker_data["open"] - 1
        
        ticker_data = ticker_data.reset_index()
        ticker_data.columns = map(str.lower, ticker_data.columns)
        ticker_data = ticker_data.drop(columns=['dividends','stock splits'])
        
        ticker_data["date"] = pd.to_datetime(ticker_data["date"])
        ticker_data.insert(1, "ticker", self.ticker)    
        
        ticker_data["range"] = (ticker_data["high"]/ticker_data["low"])-1
        
        ticker_data.dropna(how="any", inplace=True)
        ticker_data = ticker_data.reset_index(drop=True)

        
        self.history_df = pd.concat([self.all_historic_data, ticker_data])

    def get_train_test(self):
        
        train_test_split = len(self.history_df[self.history_df['date']<'2018-12-31'])

        self.train = self.history_df.loc[:train_test_split]
        self.test = self.history_df.loc[train_test_split+self.hold_length:]

    def find_best_features(self):
        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        sfs = SFS(clf, 
                k_features=self.num_features, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                cv=3)
        test_features = list(self.train.columns.drop(['date', 'ticker', 'return']))
        sfs = sfs.fit(self.train[test_features], self.train['return'])

        self.features = list(sfs.k_feature_names_)
        self.num_features = len(self.features)



    def run_sfs(self):
        
        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        sfs = SFS(clf, 
                k_features=self.k_feature_range[0], 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                cv=3)

        pipe = Pipeline([('sfs', sfs), 
                        ('clf', clf)])

        param_grid = [
        {
            'sfs__k_features': self.k_feature_range
        }
        ]
        while psutil.cpu_percent()>75:
            sleep(1)

        gs = GridSearchCV(estimator=pipe, 
                        param_grid=param_grid, 
                        scoring=self.scoring,
                        n_jobs=-1, 
                        cv=3,
                        verbose=1,
                        iid=True,
                        refit=True)

        # run gridearch
        test_features = list(self.train.columns.drop(['date', 'ticker', 'return']))

        gs = gs.fit(self.train[test_features], self.train['return'])
        
        #self.features_dict = sfs.subsets_
        """
        for i in range(len(gs.cv_results_['params'])):
            print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])
            print(gs.cv_results_['param_sfs__k_features'][i])
        print('----')
        print(gs.cv_results_.keys())
        print('----')
        """
        
        

        self.features = list(gs.best_estimator_.steps[0][1].k_feature_names_)
        self.num_features = len(self.features)
        
        """
        print('Best features:', self.features)
        print("Best parameters via GridSearch", gs.best_params_)
        print('Best score:', gs.best_score_)
        """
        


    def predict(self):
        
        model = mix.GaussianMixture(n_components=3, 
                                    covariance_type="full", 
                                    n_init=100).fit( self.train[ ['date'] + list(self.features) ].set_index("date") )

        # Predict the optimal sequence of internal hidden state
        self.test['state'] = model.predict(self.test[ ['date'] + list(self.features) ].set_index('date'))

    def get_results(self):
        #print(self.test)
        #print(self.test.groupby(by=['state'])['return'].describe())
        if self.trade_other_states:
            self.test.loc[self.test['state']==1, 'state'] = 2

        # make trades
        buy_price = None
        trades = []
        for i in range(2,len(self.test)):
            day_before_yesterday = self.test.iloc[i-2]
            yesterday = self.test.iloc[i-1]
            today = self.test.iloc[i]
            
            #if yesterday['state'] != day_before_yesterday['state'] and today['state']==2 and yesterday['state']==2 and buy_price is None:
            if today['state']==1 and yesterday['state']!=1 and buy_price is None:
                
                #print('bought\n', today)
                buy_price = float(today['close'])
                buy_date = today['date']
            elif today['state'] != 1 and buy_price is not None:
                sell_price = float(today['close'])
                sell_date = today['date']
                #print('sold\n', today)
                trades.append([buy_date, sell_date,buy_price, sell_price, sell_price / buy_price - 1])
                buy_price = None
                
        # sell if currently held
        if buy_price is not None:
            sell_price = float(today['close'])
            sell_date = None
            trades.append([buy_date, sell_date,buy_price, sell_price, sell_price / buy_price - 1])

        df = pd.DataFrame(trades, columns = ['buy_date', 'sell_date', 'buy_price', 'sell_price', 'return'])
        
        
        # todo: add abnormal change

        #df['return'].describe()
        self.result_df = df
        self.num_trades = len(df)
        if self.num_trades == 0:
            self.total_return = None
            self.accuracy = None
            self.num_trades_profitable = None
            self.trades = None
            return
        
        self.total_return = df['return'].sum()
        self.accuracy = len(df[df['return']>0])/float(len(df))
        self.num_trades_profitable = len(df[df['return']>0])

        df['ticker'] = self.ticker
        df['name'] = self.name
        self.trades = df

    
    def store_results(self):
        results = []
        results.append([self.name, str(self.features), self.hold_length, self.period, self.pattern, self.trade_other_states, self.with_decision_tree, self.with_original, self.num_features, self.max_depth, self.num_trades, self.total_return, self.accuracy])
        df = pd.DataFrame(results, columns = ['name', 'features', 'hold_length', 'period', 'pattern', 'trade_other_states', 'with_decision_tree', 'with_original', 'num_features', 'max_depth', 'num_trades', 'total_return', 'accuracy'])
        print(df)
        df.to_sql('hidden_states_test', self.conn, if_exists='append')

def run_model(params):
    
    for i in range(3):
        stock_predictor(params)
    

if __name__ == '__main__':

    param_list = []
    #results = []
    """
    for hold_length in [1,5,10]:
        for period in ['2y','3y','5y']:
            for with_decision_tree in [True, False]:
                for trade_other_states in  [True, False]:
                        for pattern in [False, True]:
                            for max_depth in [2,3,4,5]:
                                for with_original in [True,False]:
                                    if with_decision_tree==False:
                                        max_depth = None
                                        pattern = None

                                    name = namegenerator.gen()    
                                    param_list.append( {'symbol': 'QLD', 'hold_length': hold_length, 
                                                        'period': period, 'max_depth': max_depth, 
                                                        'pattern': pattern, 'trade_other_states': trade_other_states, 
                                                        'with_original': with_original, 
                                                        'with_decision_tree': with_decision_tree,
                                                        'name': name} )
    """                             
    for hold_length in [5]:
        for period in ['3y']:
            for with_decision_tree in [False]:
                for trade_other_states in  [True, False]:
                        for pattern in [False]:
                            for max_depth in [2]:
                                for with_original in [True]:
                                    if with_decision_tree==False:
                                        max_depth = None
                                        pattern = None

                                    name = namegenerator.gen()    
                                    param_list.append( {'symbol': 'QLD', 'hold_length': hold_length, 
                                                        'period': period, 'max_depth': max_depth, 
                                                        'pattern': pattern, 'trade_other_states': trade_other_states, 
                                                        'with_original': with_original, 
                                                        'with_decision_tree': with_decision_tree,
                                                        'name': name} )
                                
    p = Pool(cpu_count())
    p.map(run_model, param_list)
    #run_model(param_list[0])

"""
print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covariances_[i]))
    print()
"""

"""
sns.set(font_scale=1.5)
states = (pd.DataFrame(hidden_states, columns=['states'], index=X_test.index)
          .join(X_test, how='inner')
          .reset_index(drop=False)
          .rename(columns={'index':'Date'}))
states.head()

#suppressing warnings because of some issues with the font package
#in general, would not rec turning off warnings.
import warnings
warnings.filterwarnings("ignore")

sns.set_style('white', style_kwds)
order = [0, 1, 2]
fg = sns.FacetGrid(data=states, hue='states', hue_order=order,
                   palette=colors, aspect=1.31, height=12)
fg.map(plt.scatter, 'date', "close", alpha=0.8).add_legend()
sns.despine(offset=10)
fg.fig.suptitle('Historical SPY Regimes', fontsize=24, fontweight='demi')
"""