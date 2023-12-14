import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer, MissingIndicator
from functions import *
from optbinning import OptimalBinning, Scorecard, BinningProcess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def create_feature(df):
    # Calculate new features
    df['CREDIT_DURATION'] = df['DAYS_CREDIT'] - df['DAYS_CREDIT_ENDDATE']
    df['ENDDATE_DIF'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_ENDDATE_FACT']
    df['DEBT_PERCENTAGE'] = df['AMT_CREDIT_SUM'] / df['AMT_CREDIT_SUM_DEBT']
    df['DEBT_CREDIT_DIFF'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT']
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT_SUM'] / df['AMT_ANNUITY']
    df['BUREAU_CREDIT_FACT_DIFF'] = df['DAYS_CREDIT'] - df['DAYS_ENDDATE_FACT']
    df['BUREAU_CREDIT_ENDDATE_DIFF'] = df['DAYS_CREDIT'] - df['DAYS_CREDIT_ENDDATE']
    df['BUREAU_CREDIT_DEBT_RATIO'] = df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM']
    df['BUREAU_IS_DPD'] = df['CREDIT_DAY_OVERDUE'] > 0
    df['BUREAU_IS_DPD_OVER60'] = df['CREDIT_DAY_OVERDUE'] > 60
    df['BUREAU_IS_DPD_OVER120'] = df['CREDIT_DAY_OVERDUE'] > 120
    df['UTILIZATION_RATIO'] = df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM_LIMIT']

    # Calculate mean and rolling mean features using groupby
    df['DAYS_CREDIT_mean'] = df.groupby('SK_ID_CURR')['DAYS_CREDIT'].transform('mean')
    df['last_active_DAYS_CREDIT'] = df.groupby('SK_ID_CURR')['DAYS_CREDIT'].transform('last')
    df['BUREAU_AVG_LOAN_12M'] = df.groupby('SK_ID_CURR')['DAYS_CREDIT'].rolling(window=12).mean().reset_index(level=0, drop=True)

    # Calculate percentage of high debt ratio
    df['BUREAU_PCT_HIGH_DEBT_RATIO'] = (df['AMT_CREDIT_SUM_DEBT'] > 0.5 * df['AMT_CREDIT_SUM']).mean()

    return df

# Load data
bureau = pd.read_csv('data/raw/dseb63_bureau.csv')
bureau_balance = pd.read_csv('data/raw/dseb63_bureau_balance.csv')

# Aggregations for bureau_balance
bureau_balance = pd.get_dummies(bureau_balance, columns=['STATUS'], dummy_na=True)

bb_aggregations = bureau_balance.groupby('SK_ID_BUREAU').agg({
    'MONTHS_BALANCE': ['min', 'max', 'size', 'mean'],
    'STATUS_0': ['mean'],
    'STATUS_1': ['mean'],
    'STATUS_2': ['mean'],
    'STATUS_3': ['mean'],
    'STATUS_4': ['mean'],
    'STATUS_5': ['mean'],
    'STATUS_C': ['mean', 'count'],
    'STATUS_X': ['mean', 'count'],
    'STATUS_nan': ['mean', 'count'],
})

# Rename columns
bb_aggregations.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_aggregations.columns.tolist()])

# Create features for bureau
bureau = create_feature(bureau)

# Merge bureau_balance with bureau
bureau = bureau.merge(bb_aggregations, how='left', on='SK_ID_BUREAU')
bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
bureau.set_index('SK_ID_CURR', inplace=True)

# Replace positive inf with nan
bureau = bureau.replace([np.inf, -np.inf], np.nan)

# One-hot encoding for categorical columns with get_dummies
bureau, cat_cols = one_hot_encoder(bureau, nan_as_category= True)
print('After one-hot encoding: {}'.format(bureau.shape))

# Aggregate
bureau_agg = bureau.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'var'])
bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
bureau_agg['BUR_COUNT'] = bureau.groupby('SK_ID_CURR').size()
print('After aggregation: {}'.format(bureau_agg.shape))

# Target
target = pd.read_csv('data/interim/binned_numerical_features/target.csv')
target.set_index('SK_ID_CURR', inplace=True)
y_train = target[target.index.isin(bureau_agg.index)]['TARGET']

bureau_train = bureau_agg[bureau_agg.index.isin(target.index)]
bureau_test = bureau_agg[~bureau_agg.index.isin(target.index)]

# Drop columns with 1 unique value
cols_to_drop = [col for col in bureau_train.columns if bureau_train[col].nunique() <= 1]
bureau_train.drop(cols_to_drop, axis=1, inplace=True)
bureau_test.drop(cols_to_drop, axis=1, inplace=True)

# Binning process
variable_names = bureau_train.columns.tolist()
binning_process = BinningProcess(variable_names)
binning_process.fit(bureau_train, y_train)

# Transform train and test
bureau_train_binned = binning_process.transform(bureau_train)
bureau_train_binned.columns = [bureau_train_binned.columns[i] + '_BINNED' for i in range(len(bureau_train_binned.columns))]
bureau_train_binned.index = bureau_train.index
bureau_test_binned = binning_process.transform(bureau_test)
bureau_test_binned.columns = [bureau_test_binned.columns[i] + '_BINNED' for i in range(len(bureau_test_binned.columns))]
bureau_test_binned.index = bureau_test.index

# Merge original with binned
bureau_train_binned = pd.concat([bureau_train, bureau_train_binned], axis=1)
bureau_test_binned = pd.concat([bureau_test, bureau_test_binned], axis=1)

# Select features by IV
print('Selecting features...')
selected_features = select_features_iv(bureau_train_binned, y_train, threshold=0.02)
print(f'Number of selected features: {len(selected_features)}')
bureau_train = bureau_train_binned[selected_features]
bureau_test = bureau_test_binned[selected_features]

# Concat train and test
bureau = pd.concat([bureau_train, bureau_test], axis=0)

# Save
bureau.to_csv('data/interim/binned_numerical_features/processed_bureau.csv')
