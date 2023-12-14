import pandas as pd
import joblib

def main():
    # Load test data
    test_df = pd.read_csv('data/processed/test_df.csv')

    # Load trained model and preprocessor
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')

    # Prepare test data
    X_test = test_df.drop('TARGET', axis=1)
    X_test = preprocessor.transform(X_test)

    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Prepare submission
    submission = pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'], 'TARGET': y_pred_proba})
    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype('int32')
    submission.set_index('SK_ID_CURR', inplace=True)
    submission.to_csv('submission.csv')

if __name__ == "__main__":
    main()
