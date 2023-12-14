import time
import lightgbm as lgbm
import pandas as pd

class timer:
    def __init__(self, title):
        self.title = title

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print("{} - done in {:.0f}s".format(self.title, elapsed_time))

def all_data_split(all_data):
    
    all_data.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in all_data.columns]
    
    train_df = all_data[all_data['TARGET'].notnull()]
    test_df = all_data[all_data['TARGET'].isnull()]
    print(all_data['TARGET'].notnull().sum())
    print(all_data['TARGET'].isnull().sum())
    return train_df, test_df


def select_features_lightgbm(X, y, threshold=0.001):
    """
    Select features using LightGBM.
    Parameters:
    X (DataFrame): The input DataFrame.
    y (Series): The target variable.
    threshold (float): The threshold for feature selection.
    Returns:
    DataFrame: The DataFrame containing feature importances.
    """
    cols = X.columns
    lgbm_model = lgbm.LGBMClassifier()
    lgbm_model.fit(X, y)
    importances = pd.Series(lgbm_model.feature_importances_, index=cols)
    # scale by max
    importances = importances / importances.max()
    return importances[importances >= threshold]

def modeling(all_data):
    
    all_data.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in all_data.columns]
    
    train_df = all_data[all_data['TARGET'].notnull()]
    test_df = all_data[all_data['TARGET'].isnull()]
    print(all_data['TARGET'].notnull().sum())
    print(all_data['TARGET'].isnull().sum())
    return train_df, test_df