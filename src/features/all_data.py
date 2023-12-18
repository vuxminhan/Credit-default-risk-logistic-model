import pandas as pd
import pickle
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from src.helpers.utils import modeling


def main():
    all_data = pd.read_csv('data/interim/simple_aggregates/all_data.csv')
    
    lag_features = pd.read_csv('data/interim/lead_lag/lag_features.csv')
    all_data = all_data.merge(lag_features, on='SK_ID_CURR', how='left')
    
    nested_features = pd.read_csv('data/interim/nested_model/pred_stats.csv')
    all_data = all_data.merge(nested_features, on = ['SK_ID_CURR'], how = 'left')
    
    all_data.replace('', np.nan, inplace=True)
    all_data.head() 
    
    binned_features = pd.read_csv('data/interim/binned_numerical_features/processed_data.csv')
    all_data = all_data.merge(binned_features, on='SK_ID_CURR', how='left')
    train_df,test_df = modeling(all_data)
    
    slicing_test = pd.read_csv("/kaggle/input/test-62-feats/new_feat_test.csv")
    slicing_train = pd.read_csv("/kaggle/input/test-62-feats/new_feat_train.csv")
    test_df = test_df.merge(slicing_test, on = ['SK_ID_CURR'], how = 'left')
    train_df = train_df.merge(slicing_train, on = ['SK_ID_CURR'], how = 'left')
    
    #concat train and test
    all_df = pd.concat([train_df,test_df],axis=0)
    all_df.to_csv('data/processed/all_data.csv', index=False)
    
if __name__ == "__main__":
    main()