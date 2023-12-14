import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import os
import sys
sys.path.append(os.getcwd())
from src.helpers.utils import modeling, select_features_lightgbm


class InfinityToNanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace([np.inf, -np.inf], np.nan)

def build_preprocessor(numerical_cols, categorical_cols):
    numerical_transformer = make_pipeline(
        InfinityToNanTransformer(),
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor

def main():
    # # Load and preprocess data
    # with open('data/processed/all_data.pkl', 'rb') as f:
    #     all_data = pickle.load(f)
    
    all_data = pd.read_csv('data/processed/all_data.csv')
    train_df, test_df = modeling(all_data)
    test_df.to_csv('data/processed/test_df.csv', index=False)
    X = train_df.drop('TARGET', axis=1)
    y = train_df['TARGET']
    
    # Feature selection
    selected_features = select_features_lightgbm(X, y, threshold=0.005)
    X = X[selected_features.index]
    numerical_cols = X.select_dtypes(include='number').columns
    categorical_cols = X.select_dtypes(include='object').columns

    # Define preprocessing and modeling pipeline
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression())])

    # Training and cross-validation
    cv_results = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print("ROC AUC scores for each fold:", cv_results)
    print("Average ROC AUC:", cv_results.mean())
    print("Gini Coefficient:", 2 * cv_results.mean() - 1)

    # Train the final model
    model.fit(X, y)

    # Save the model and preprocessor
    joblib.dump(model, 'model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')

if __name__ == "__main__":
    main()
