import sqlite3
import pandas as pd

conn = sqlite3.connect('tiny_pipeline.db')

sql = 'select features, moderate_return from models group by features order by moderate_return desc limit 10'
df = pd.read_sql(sql, conn)

all_features_used = []
for key, value in df.iterrows():
    features = eval(df.loc[key, 'features'])
    all_features_used.extend(features)
    """
    features = sorted(features)
    for i in range(len(features)):

        df.loc[key,'feature_'+str(i)] = features[i]
    """
features_used = list(set(all_features_used))
print(features_used)
input()
for feature in features_used:
    df[feature] = 0
    for key, values in df.iterrows():
        if feature in df.loc[key, 'features']:
            df.loc[key, feature] = 1

print(df)

input()

result_features = []
for feature in features_used:
    
    group = list(df.groupby(by=feature))[1]
    
    group = group[1]['moderate_return']
    
    result_features.append([feature, len(group), group.mean()])

result_df = pd.DataFrame(result_features, columns = ['features', 'num_in_group', 'means'])
result_df = result_df.sort_values(by=['means'])
print(result_df)
overall_average = result_df['means'].mean()
best_features = sorted( result_df[result_df['means']>overall_average]['features'].values )
print(best_features)
#result_df.to_csv('best_features.csv')