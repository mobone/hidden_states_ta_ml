import yfinance
from ta_indicators import get_ta
import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_extraction import PrincipalComponentAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from random import randint
import numpy as np
import warnings
warnings.simplefilter("ignore")
def get_data(symbol):
        n_components = 3
        
        history = yfinance.Ticker(symbol).history(period='25y', auto_adjust=False).reset_index()
        
        history = get_ta(history, volume=True, pattern=False)
        history.columns = map(str.lower, history.columns)
        
        history['date'] = pd.to_datetime(history['date'])
        history['return'] = history['close'].pct_change() * 100
        history = history.dropna()
        
        history.loc[history['high']<history['open'], 'high'] = history['open']+.01

        history['next_return'] = history['return'].shift(-1)


        # run decision tree to find best features
        test_cols = list(history.columns.drop(['date','return', 'next_return']))
        X_train, X_test, y_train, y_test = train_test_split(history[ test_cols ], history['return'], test_size=0.33, random_state=42)
        
        
        clf = ExtraTreesRegressor(n_estimators=150)
        clf = clf.fit(X_train, y_train)
        df = pd.DataFrame([test_cols, clf.feature_importances_]).T
        df.columns = ['feature', 'importances']
        
        df = df.sort_values(by='importances')
        print(df)
        
        starting_features = list(df['feature'].tail(20).values)

        total_train = history.head( 252 * 20 )

        found_cutoffs = {}
        found_cutoffs['large_mean_distance'] = [-np.inf,None, None]
        found_cutoffs['small_mean_distance'] = [np.inf,None, None]
        found_cutoffs['large_positive_mean'] = [-np.inf,None, None]
        found_cutoffs['small_positive_mean'] = [np.inf,None, None]
        found_cutoffs['large_negative_mean'] = [-np.inf,None, None]
        found_cutoffs['small_negative_mean'] = [np.inf,None, None]
        found_cutoffs['large_var_distance'] = [-np.inf,None, None]
        found_cutoffs['small_var_distance'] = [np.inf,None, None]
        found_cutoffs['large_var_std'] = [-np.inf,None, None]
        found_cutoffs['small_var_std'] = [np.inf,None, None]
        i = 0
        while i<1520:
            pipe_pca = make_pipeline(StandardScaler(),
                            PrincipalComponentAnalysis(n_components=n_components),
                            GMMHMM(n_components=n_components, covariance_type='full', n_iter=150, random_state=7),
                            )

            random_start = int(randint(0, len(total_train)))
            random_length = int(randint(3,25)*20)
            try:
                test_period = total_train.iloc[random_start:random_start+random_length]
                pipe_pca.fit( test_period[ starting_features ] )
                test_period['state'] = pipe_pca.predict( test_period[ starting_features ] )
            except:
                continue
            #print(test_period)
            #print(random_start, random_length)

            results = pd.DataFrame()
            for key, group in test_period.groupby(by='state'):
                results.loc[key, 'mean'] = group['return'].mean()
                results.loc[key, 'var'] = group['return'].std()
            results = results.sort_values(by='mean')
            #print(results)

            distance_between_means = abs(float(results.iloc[0]['mean'])) + float(results.iloc[2]['mean'])
            # biggest distance between means
            if distance_between_means > found_cutoffs['large_mean_distance'][0]:
                found_cutoffs['large_mean_distance'][0] = distance_between_means
                found_cutoffs['large_mean_distance'][1] = random_start
                found_cutoffs['large_mean_distance'][2] = random_length

            # smallest distance between means
            if distance_between_means < found_cutoffs['small_mean_distance'][0]:
                found_cutoffs['small_mean_distance'][0] = distance_between_means
                found_cutoffs['small_mean_distance'][1] = random_start
                found_cutoffs['small_mean_distance'][2] = random_length
            
            positive_mean = float(results.iloc[2]['mean'])
            # largest positive state mean
            if positive_mean > found_cutoffs['large_positive_mean'][0]:
                found_cutoffs['large_positive_mean'][0] = positive_mean
                found_cutoffs['large_positive_mean'][1] = random_start
                found_cutoffs['large_positive_mean'][2] = random_length

            # smallest positive state mean
            if positive_mean < found_cutoffs['small_positive_mean'][0]:
                found_cutoffs['small_positive_mean'][0] = positive_mean
                found_cutoffs['small_positive_mean'][1] = random_start
                found_cutoffs['small_positive_mean'][2] = random_length

            
            negative_mean = float(results.iloc[0]['mean'])
            # largest negative state mean
            if negative_mean > found_cutoffs['large_negative_mean'][0]:
                found_cutoffs['large_negative_mean'][0] = negative_mean
                found_cutoffs['large_negative_mean'][1] = random_start
                found_cutoffs['large_negative_mean'][2] = random_length

            # smallest negative state mean
            if negative_mean < found_cutoffs['small_negative_mean'][0]:
                found_cutoffs['small_negative_mean'][0] = negative_mean
                found_cutoffs['small_negative_mean'][1] = random_start
                found_cutoffs['small_negative_mean'][2] = random_length


            results = results.sort_values(by='var')
            distance_between_vars = abs(float(results.iloc[0]['var'])) + float(results.iloc[2]['var'])
            # largest distance between variations
            if distance_between_vars > found_cutoffs['large_var_distance'][0]:
                found_cutoffs['large_var_distance'][0] = distance_between_vars
                found_cutoffs['large_var_distance'][1] = random_start
                found_cutoffs['large_var_distance'][2] = random_length

            # smallest distance between variations
            if distance_between_vars < found_cutoffs['small_var_distance'][0]:
                found_cutoffs['small_var_distance'][0] = distance_between_vars
                found_cutoffs['small_var_distance'][1] = random_start
                found_cutoffs['small_var_distance'][2] = random_length

            # largest average variation
            variation_std = results['var'].std()
            if variation_std > found_cutoffs['large_var_std'][0]:
                found_cutoffs['large_var_std'][0] = variation_std
                found_cutoffs['large_var_std'][1] = random_start
                found_cutoffs['large_var_std'][2] = random_length

            # smallest average variation
            if variation_std < found_cutoffs['small_var_std'][0]:
                found_cutoffs['small_var_std'][0] = variation_std
                found_cutoffs['small_var_std'][1] = random_start
                found_cutoffs['small_var_std'][2] = random_length


            i = i + 1
            if i % 100 == 1:
                df = pd.DataFrame.from_dict(found_cutoffs, orient='index', columns = ['score', 'start', 'length'])
                print(df)
                df.to_csv('best_models.csv')
            

get_data('SPY')

