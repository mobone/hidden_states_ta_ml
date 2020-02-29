from utils import get_data, get_industry_tickers
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn import mixture as mix
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from math import floor, ceil
import yfinance
import sqlite3

class model_generator():
    def __init__(self, data, scoring):
        # TODO: test different time scales
        # TODO: make class return results for multiple years, check for consistency 
        conn = sqlite3.connect('markov_models.db')
        self.data = data
        # iterate through these params
        self.scoring = scoring
        self.k_features = 6
        self.max_depth = 6
        
        self.test_dates = ('2017-01-01','2020-02-28')

        #self.get_train_test()

        for self.k_features in range(1,11):
            for self.max_depth in range(1,11):
                for self.scoring in ['r2', 'neg_mean_squared_error']:
                    
                    self.get_train_test()
                    # maybe play with these params
                    #self.covariance_type = ['diag','full','tied']
                    #self.n_components = 3

                    
                    self.run_decision_tree()
                    for self.with_original in [False, True]:  
                        self.get_train_test()  
                        if self.with_original:
                            self.features = list(set(self.features + ['range', 'close']))
                        print(self.features)
                        
                        self.generate_markov_model()
                        self.get_results()
                        self.result_df['k_features'] = self.k_features
                        self.result_df['max_depth'] = self.max_depth
                        self.result_df['scoring'] = self.scoring
                        self.result_df['with_original'] = self.with_original
                        self.result_df['features'] = str(self.features)
                        print(self.result_df)
                        self.result_df.to_sql('models', conn, if_exists='append', index=False)
                        
                        #self.simple_trader()
                        #self.walk_timeline()
                        #self.plot_results()

    def get_train_test(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        #self.data['date'] = self.data['date'].astype(str)
        self.train = self.data[self.data['date']<self.test_dates[0]]
        self.test = self.data[ (self.data['date']>self.test_dates[0]) & (self.data['date']<self.test_dates[1]) ]
        #print(self.train)
        #print(self.test)

    def run_decision_tree(self):    
        clf = DecisionTreeRegressor(random_state=7, max_depth=self.max_depth)
        
        sfs = SFS(clf, 
                k_features=self.k_features, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=-1,
                cv=4)
        test_features = self.train.columns.copy()
        test_features = list(test_features.drop(['date', 'ticker', 'return']))

        sfs = sfs.fit(self.train[test_features], self.train['return'])
        
        self.score = sfs.k_score_
        
        self.features = list(sfs.k_feature_names_)
        print('features', self.features)
        

    def run_grid_search(self):
        # find best params and features, using return as the target
        clf = DecisionTreeRegressor(random_state=7, max_depth=4)
        sfs = SFS(clf, 
                k_features=4, 
                forward=True, 
                floating=True, 
                scoring=self.scoring,
                n_jobs=1,
                cv=4)

        pipe = Pipeline([('sfs', sfs), 
                        ('clf', clf)])
        
        param_grid = [{
           'sfs__k_features': list(range(2,5)),
            'sfs__estimator__max_depth': list(range(3,6))
        }]

        gs = GridSearchCV(estimator=pipe, 
                        param_grid=param_grid, 
                        scoring=self.scoring, 
                        n_jobs=15, 
                        cv=4,
                        verbose=1,
                        iid=True,
                        refit=True)

        test_features = list(self.train.columns.drop(['date', 'ticker', 'return']))

        gs = gs.fit(self.train[test_features], self.train['return'])

        self.score = gs.best_score_
        
        self.features = list(gs.best_estimator_.steps[0][1].k_feature_names_)

        print('params', gs.best_params_)
        print('features', self.features)
        print('score', self.score)

        self.k_features = gs.best_params_['sfs__k_features']
        self.max_depth = gs.best_params_['sfs__estimator__max_depth']
    
    def generate_markov_model(self):
        # use percent return and try to minimize variance and maximize return
                # TODO: generate model every day
        self.model = mix.GaussianMixture(n_components=3, 
                                    covariance_type="full", #tied
                                    random_state=7,
                                    n_init=100)
        self.train = self.train.set_index('date')
        self.model.fit( self.train[ ['return'] + self.features] )

        self.test = self.test.set_index('date')
        self.test['state'] = self.model.predict(self.test[['return'] + self.features])

        # TODO rename state with english text
        #self.test['close_shifted'] = self.test['close'].shift(-1)
        
        #self.hidden_states = self.model.predict(self.test[['return'] + self.features])
        #self.test[['date']+self.features+['open','close','state']].to_csv('test.csv')

    def get_results(self):
        print("Means and vars of each hidden state")
        results = []
        for i in range(self.model.n_components):
            results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0] ])
            #results.append([ i, self.model.means_[i][0], np.diag(self.model.covariances_[i])[0][0] ])
            
        result_df = pd.DataFrame(results, columns = ['state','mean', 'var'])
        result_df = result_df.set_index('state')
        
        result_df = result_df.sort_values(by=['mean'])
        # todo: figure out what to do with intermediate state
        self.good_state = int(result_df.tail(1).index.values[0])
        self.middle_state = int(result_df.tail(2).head(1).index.values[0])
        self.bad_state = int(result_df.head(1).index.values[0])
        """
        print(self.features)
        print()
        print(result_df)
        print('good_state', self.good_state, 'bad_state', self.bad_state)
        print()
        """

        # TODO: CLEAN THIS UP

        good_result = result_df.loc[self.good_state]
        middle_result = result_df.loc[self.middle_state]
        bad_result = result_df.loc[self.bad_state]
        
        good_result.index = ['good_mean', 'good_var']
        middle_result.index = ['middle_mean', 'middle_var']
        bad_result.index = ['bad_mean', 'bad_var']

        good_result=good_result.rename('0')
        middle_result=middle_result.rename('0')
        bad_result=bad_result.rename('0')

        result_df = pd.concat([good_result, middle_result, bad_result])
        result_df = pd.DataFrame(result_df).T
        result_df['good_state'] = self.good_state
        result_df['middle_state'] = self.middle_state
        result_df['bad_state'] = self.bad_state
        result_df['distance'] = result_df['good_mean'] + (result_df['bad_mean'] * -1)
        self.result_df = result_df


    def simple_trader(self):
        self.test['percent_change'] = self.test['close'].shift(-1) / self.test['close'] - 1

        for key, df in self.test.groupby(by='state'):
            print(key, round(df['percent_change'].mean(),4), round(df['percent_change'].std(),4), len(df['percent_change']))

    def plot_results(self):
        sns.set(font_scale=1.25)
        style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,'legend.frameon': True}
        sns.set_style('white', style_kwds)

        colors = cm.rainbow(np.linspace(0, 1, self.model.n_components))
        
        """
        fig, axs = plt.subplots(self.model.n_components, sharex=True, sharey=True, figsize=(12,9))

        for i, (ax, color) in enumerate(zip(axs, colors)):
            # Use fancy indexing to plot data in each state.
            mask = self.hidden_states == i
            ax.plot_date(self.test.index.values[mask],
                        self.test["close"].values[mask],
                        ".-", c=color)
            ax.set_title("{0}th hidden state".format(i), fontsize=16, fontweight='demi')

            # Format the ticks.
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_minor_locator(MonthLocator())
            sns.despine(offset=10)

        plt.tight_layout()
        #plt.show()
        """
        print(self.test['state'].values)
        #self.test.loc[self.test['state']==2, 'state'] = 1
        sns.set(font_scale=1.5)
        states = (pd.DataFrame(self.test['state'].values, columns=['states'], index=self.test.index)
                .join(self.test, how='inner')
                .reset_index(drop=False)
                .rename(columns={'index':'Date'}))
        print(states.tail(100))

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
        plt.show()
    
    def walk_timeline(self):
        # todo: use middle state for regular index and best state for bonus index
        tqqq_history = yfinance.Ticker('TQQQ').history(period='5y', auto_adjust=False)
        tqqq_history = tqqq_history.reset_index()
        print(self.test.head(1).index.values[0])
        tqqq_history = tqqq_history[tqqq_history['Date']>=self.test.head(1).index.values[0]]
        tqqq_history = tqqq_history.iloc[:-1]
        
        total_results = []

        for share_accumulation_rate in range(1,10):
            share_accumulation_rate = share_accumulation_rate * 5
            for share_decay_rate in range(1,10):
                
                share_decay_rate = share_decay_rate * 5
                #share_accumulation_rate = 3
                num_shares = 0
                #max_shares = 50
                
                keep_for_bonus = 2000
                balance = 10000

                starting_balance = 10000 + keep_for_bonus

                balances = []

                held_shares = {}
                held_shares['QQQ'] = {'num_shares': 0}
                held_shares['TQQQ'] = {'num_shares': 0}
                """
                print(self.test.head(1))
                print(self.test.tail(1))
                print(tqqq_history.head(1))
                print(tqqq_history.tail(1))
                input()
                """
                buy_and_hold_start = balance + keep_for_bonus
                benchmark_return = balance * float(self.test['close'].tail(1)) / float(self.test['close'].head(1))
                tqqq_return = keep_for_bonus * float(tqqq_history['Close'].tail(1)) / float(tqqq_history['Close'].head(1))
                
                self.test = self.test[   list(set(['close']+self.features+['state']))  ]
                for i in self.test.index:
                    today = self.test.loc[i]
                    today_bonus = tqqq_history.loc[tqqq_history['Date'] == i]
                    bonus_share_price = float(today_bonus['Close'])
                    share_price = float(today['close'])
                    #print(pd.DataFrame(today).T)

                    
                    if (today['state']==self.good_state) and (balance - (share_accumulation_rate*share_price) > 0): # and held_shares['QQQ']['num_shares']<max_shares:
                        # buy more shares if we can
                            #print('buying %s shares at %s for a cost of %s' % ( share_accumulation_rate, share_price, round(share_accumulation_rate * share_price, 2)))
                            held_shares['QQQ']['num_shares'] = held_shares['QQQ']['num_shares'] + share_accumulation_rate
                            balance = balance - round( (share_accumulation_rate*share_price), 2)
                    elif (today['state']==self.good_state) and (balance - (share_accumulation_rate*share_price) < 0) and held_shares['TQQQ']['num_shares']==0:
                        # use bonus if we can't buy more regular shares
                        #print(keep_for_bonus)
                        #print(bonus_share_price)
                        bonus_shares = floor(keep_for_bonus / float(bonus_share_price))
                        held_shares['TQQQ']['num_shares'] = bonus_shares
                        keep_for_bonus = keep_for_bonus - (bonus_shares * float(bonus_share_price))

                    elif today['state']==self.bad_sate:
                        
                        num_shares_after_selling = ceil(held_shares['QQQ']['num_shares']/share_decay_rate)
                        num_shares_sold = held_shares['QQQ']['num_shares'] - num_shares_after_selling
                        #print('selling %s shares, down to %s shares' % ( num_shares_sold, num_shares_after_selling ))
                        held_shares['QQQ']['num_shares'] = num_shares_after_selling
                        balance = balance + round( (num_shares_sold * share_price), 2)

                        if held_shares['TQQQ']['num_shares']!=0:
                            keep_for_bonus = keep_for_bonus + (held_shares['TQQQ']['num_shares'] * float(bonus_share_price))
                            held_shares['TQQQ']['num_shares'] = 0

                    

                    held_balance = 0
                    #for key in held_shares.keys():
                    held_balance = held_balance + round( (held_shares['QQQ']['num_shares'] * float(share_price)), 2 )
                    held_balance = held_balance + round(  held_shares['TQQQ']['num_shares'] * float(bonus_share_price), 2)
                    balances.append( [today['state'], balance, keep_for_bonus, held_balance, balance + keep_for_bonus + held_balance, held_shares['QQQ']['num_shares'], held_shares['TQQQ']['num_shares']] )
                    #print(pd.DataFrame(balances, columns = ['state', 'bank', 'bonus', 'held_balance', 'total', 'held_shares', 'held_bonus_shares']))
                    
                    
                
                ending_balance = balance + keep_for_bonus + round( (held_shares['QQQ']['num_shares'] * share_price), 2) + round( (held_shares['TQQQ']['num_shares'] * bonus_share_price), 2)
                #print(ending_balance)
                buy_and_hold_percent = (benchmark_return + tqqq_return) / buy_and_hold_start - 1
                
                total_results.append([share_accumulation_rate, share_decay_rate, round(ending_balance, 2), round((ending_balance / starting_balance - 1),4), round(buy_and_hold_percent, 4)])
                result_df = pd.DataFrame(total_results, columns = ['accum','decay', 'ending balance', 'percent', 'benchmark'])
                print(result_df)
                print(result_df['percent'].max())
        result_df.to_csv('test.csv')
                
        input()
            
            
            

            
            #self.test.loc[i, 'num_shares'] = num_shares
        #self.test[ ['state', 'close', 'close_shifted', 'num_shares'] ].to_csv('test.csv')


#tickers = get_industry_tickers(sector='technology')[:25]
#tickers.remove('GOOGL')
tickers = ['QQQ']

data = get_data(tickers,period='max',pattern=False)

scoring = 'neg_mean_squared_error'
model_generator(data, scoring)