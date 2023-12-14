import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import time
import glob
from joblib import Parallel, delayed
import pickle
sns.set_style('white')


DATA_DIR = 'data/raw/'

def get_path(str, first=True, parent_dir='./**/'):
    res_li = glob.glob(parent_dir+str)
    return res_li[0] if first else res_li

def load_folds_lables():
    path = 'data/interim/hrdr-prepare-kfold/'
    eval_sets = np.load(path+'eval_sets.npy')
    y = np.load(path+'target.npy')
    return eval_sets, y


def nest_print(dict_item, inline=True, indent=True):
    s = []
    s_ind = '\t' if indent else ''
    for k, v in dict_item.items():
        s += [': '.join([str(k), str(round(v, 6))])]
    if inline:
        print(s_ind+' '.join(s))
    else:
        print(s_ind+'\n'.join(s))
   
def label_encoding(df):
    obj_cols = [c for c in df.columns if df[c].dtype=='O']
    for c in obj_cols:
        df[c] = pd.factorize(df[c], na_sentinel=-1)[0]
    df[obj_cols].replace(-1, np.nan, inplace=True)
    return df

def logistic_cv_train(
    name, params, X, y, X_test, 
    num_folds, metric=roc_auc_score,
    verbose_cv=True, msgs={}
):
    pred_test = np.zeros((X_test.shape[0],))
    pred_val = np.zeros((X.shape[0],))
    cv_scores = []
    models = []
    kfolds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    for train_index, valid_index in kfolds.split(X, y):
        print('[level 1] processing fold...')
        t0 = time.time()

        # Split data
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # Fit model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Predictions
        pred_val[valid_index] = model.predict_proba(X_valid)[:, 1]
        pred_test += model.predict_proba(X_test)[:, 1] / num_folds

        # Evaluate
        scr = metric(y_valid, pred_val[valid_index])
        if verbose_cv:
            print(f'{name} auc:', scr, 
                  f'fold done in {time.time() - t0:.2f} s')
        cv_scores.append(scr)
        models.append(model)

    msgs = dict(
        msgs, 
        cv_score_mean=np.mean(cv_scores), 
        cv_score_std=np.std(cv_scores),
        cv_score_min=np.min(cv_scores), 
        cv_score_max=np.max(cv_scores),
    )
    nest_print(msgs)

    result = dict(
        name=name,
        pred_val=pred_val,
        pred_test=pred_test,
        cv_scores=cv_scores,
        models=models
    )
    return result

def preprocess_data(df):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def calc_stats(key, group):
    res = pd.Series()
    res['SK_ID_CURR'] = key
    res['_max'] = np.max(group)
    res['_min'] = np.min(group)
    res['_median'] = np.median(group)
    res['_mean'] = np.mean(group)
    res['_std'] = np.std(group)
    res['_size'] = np.size(group)
    res['_sum'] = np.sum(group)
    res['_skew'] = group.skew()
    res['_kurtosis'] = group.kurtosis()
    return res

folds, labels = load_folds_lables()
nfolds = 5
train_num = len(labels)

train_ids = pd.read_csv(DATA_DIR + 'dseb63_application_train.csv', usecols=['SK_ID_CURR'])['SK_ID_CURR'].values
test_ids = pd.read_csv(DATA_DIR +'dseb63_application_test.csv', usecols=['SK_ID_CURR'])['SK_ID_CURR'].values
sk_id_curr = np.load('data/interim/hrdr-prepare-kfold/sk_id_curr.npy')

id_labels = pd.DataFrame()
id_labels['SK_ID_CURR'] = sk_id_curr
id_labels['TARGET'] = -1
id_labels['TARGET'][:train_num] = labels
id_labels['fold'] = -1
id_labels['fold'][:train_num] = folds

results = {}

logistic_params = {
    'penalty': 'l2', # Regularization type: 'l1', 'l2', 'elasticnet', or 'none'
    'C': 1.0,        # Inverse of regularization strength; smaller values specify stronger regularization
    'solver': 'liblinear', # Algorithm to use in the optimization problem: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    'max_iter': 100, # Maximum number of iterations taken for the solvers to converge
    'random_state': 42 # Seed of the pseudo random number generator
}
num_folds = 5 

csvname_li = [
    'dseb63_bureau',
    #'bureau_balance',
    'dseb63_previous_application',
    'dseb63_installments_payments',
    'dseb63_credit_card_balance',
    'dseb63_POS_CASH_balance',
]

for csvname in csvname_li:
    print(f'Current: {csvname}...')
    
    df = pd.read_csv(DATA_DIR+f'{csvname}.csv')
    df = df.loc[np.isin(df['SK_ID_CURR'], sk_id_curr)]
    df = df.merge(id_labels, how='left', on='SK_ID_CURR')
    df = label_encoding(df)
    df = preprocess_data(df)
    
    eval_cols = ['SK_ID_CURR', 'fold', 'TARGET']
    eval_df = df[eval_cols].copy()
    df.drop(eval_cols, axis=1, inplace=True)

    feature_name = df.columns.tolist()
    y = eval_df['TARGET'].values.copy()
    X = df.loc[y!=-1].values
    X_test = df.loc[y==-1].values
    cv_folds = eval_df.loc[y!=-1, 'fold'].values
    y = y[y!=-1]

    print('shapes', X.shape, y.shape, cv_folds.shape, X_test.shape)

    results[csvname] = logistic_cv_train(
        f'{csvname}', logistic_params,
        X, y, X_test,
        num_folds=num_folds
    )

    eval_df['pred'] = -1
    eval_df.loc[eval_df['fold']!=-1, 'pred'] = results[csvname]['pred_val']
    eval_df.loc[eval_df['fold']==-1, 'pred'] = results[csvname]['pred_test']
    results[csvname]['eval_df'] = eval_df.copy()

stat_func_li = ['size', 'sum', 'min', 'median', 'max', 'mean', 'std', 'skew', 'kurtosis']

with open('mr_results.pkl', 'wb') as f:
    pickle.dump(results, f)
    
pred_stats = pd.DataFrame()
pred_stats['SK_ID_CURR'] = sk_id_curr
for csvname in csvname_li:
    print(f'Calculating pred stats of {csvname}')
    grp = results[csvname]['eval_df'].groupby('SK_ID_CURR')
    res = Parallel(n_jobs=-1)(
        delayed(calc_stats)(key, group) for key, group in tqdm(grp['pred'], total=len(grp.groups))
    )
    res = pd.concat(res, axis=1).T
    res['SK_ID_CURR'] = res['SK_ID_CURR'].astype('int')
    res.columns = [csvname+c if c!='SK_ID_CURR' else c  for c in res.columns]
    pred_stats = pred_stats.merge(res, on='SK_ID_CURR', how='left')


pred_stats.to_csv('data/processed/pred_stats.csv', index=False)
