import pandas as pd

# Load datasets
train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

# Select columns and preprocess
columns = ['SK_ID_CURR', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CODE_GENDER', 'REGION_POPULATION_RELATIVE', 'TARGET']
test['TARGET'] = None

train2 = train[columns]
test2 = test[columns[:-1]]

for df in [train2, test2]:
    df.loc[:, 'DAYS_REGISTRATION'] = df['DAYS_REGISTRATION'] - df['DAYS_BIRTH']
    df.loc[:, 'DAYS_ID_PUBLISH'] = df['DAYS_ID_PUBLISH'] - df['DAYS_BIRTH']

# Combine train and test datasets
dfull2 = pd.concat([train2, test2], axis=0)

# Identify duplicate groups
group_columns = ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CODE_GENDER', 'REGION_POPULATION_RELATIVE']
dupes = dfull2.groupby(group_columns).size().reset_index(name='n')
dupes = dupes[dupes['n'] > 1].sort_values(by='n', ascending=False)
dupes['cluster'] = range(1, len(dupes) + 1)
dupes.drop(columns='n', inplace=True)

# Merge and create lag/lead features
dfull2 = dfull2.merge(dupes, on=group_columns, how='left')
dfull2.sort_values(by=['cluster', 'DAYS_BIRTH'], inplace=True)

dfull2['lag_TARGET'] = dfull2.groupby('cluster')['TARGET'].shift(-1)
dfull2['lead_TARGET'] = dfull2.groupby('cluster')['TARGET'].shift(1)

# Handle missing values
dfull2.loc[dfull2['cluster'].isna(), ['lag_TARGET', 'lead_TARGET']] = None

lag_agg = dfull2.groupby('lag_TARGET').agg({'lag_TARGET': 'size', 'TARGET': ['mean', lambda x: x.isna().sum()]}).reset_index()
lag_agg.columns = ['lag_TARGET', 'n', 'mean_target', 'test_count']

lead_agg = dfull2.groupby('lead_TARGET').agg({'lead_TARGET': 'size', 'TARGET': ['mean', lambda x: x.isna().sum()]}).reset_index()
lead_agg.columns = ['lead_TARGET', 'n', 'mean_target', 'test_count']
# Optionally, you can print or export these aggregations
lag_features = dfull2[['SK_ID_CURR', 'lag_TARGET']]
lag_features.to_csv('data/interim/lead_lag/lag_features.csv', index=False)