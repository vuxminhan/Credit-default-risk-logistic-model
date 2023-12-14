import pandas as pd
import numpy as np
from functions import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from optbinning import BinningProcess

# Load data
train = pd.read_csv('data/interim/binned_numerical_features/application_train.csv')
test = pd.read_csv('data/interim/binned_numerical_features/application_test.csv')
target = pd.read_csv('data/interim/binned_numerical_features/target.csv')

train.set_index('SK_ID_CURR', inplace=True)
train.sort_index(inplace=True)
test.set_index('SK_ID_CURR', inplace=True)
test.sort_index(inplace=True)
target.set_index('SK_ID_CURR', inplace=True)
target = target['TARGET']

print(f'Train shape: {train.shape}, Test shape: {test.shape}, Target shape: {target.shape}')

# Merge train and test
train['is_train'] = 1
test['is_train'] = 0
data = pd.concat([train, test], axis=0)

# Merge with previous application
previous_application = pd.read_csv('data/interim/binned_numerical_features/processed_previous_application.csv')
print(f'Previous application shape: {previous_application.shape}')
data = data.merge(previous_application, how='left', on='SK_ID_CURR')

# Merge with credit card balance
credit_card_balance = pd.read_csv('data/interim/binned_numerical_features/processed_credit_card_balance.csv')
print(f'Credit card balance shape: {credit_card_balance.shape}')
data = data.merge(credit_card_balance, how='left', on='SK_ID_CURR')

# Merge with installments payments
installments_payments = pd.read_csv('data/interim/binned_numerical_features/processed_installments.csv')
print(f'Installments payments shape: {installments_payments.shape}')
data = data.merge(installments_payments, how='left', on='SK_ID_CURR')

# Merge with bureau
bureau = pd.read_csv('data/interim/binned_numerical_features/processed_bureau.csv')
print(f'Bureau shape: {bureau.shape}')
data = data.merge(bureau, how='left', on='SK_ID_CURR')

# Merge with pos cash balance
pos_cash_balance = pd.read_csv('data/interim/binned_numerical_features/processed_pos_cash.csv')
print(f'POS cash balance shape: {pos_cash_balance.shape}')
data = data.merge(pos_cash_balance, how='left', on='SK_ID_CURR')


# Print shape after merge
print(f'Merged data shape: {data.shape}')

# Drop duplicate columns
data = data.loc[:, ~data.columns.duplicated()]

# Replace inf with nan
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop target
data.drop(['TARGET'], axis=1, inplace=True, errors='ignore')

# Set index
data.set_index('SK_ID_CURR', inplace=True)

data.to_csv('data/interim/binned_numerical_features/binned_data.csv')