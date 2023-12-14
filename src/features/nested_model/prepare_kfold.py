
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import feather

DATA_DIR = 'data/raw/'
nrows = None
usecols = None
read_params = dict(nrows=nrows, usecols=usecols)
train = pd.read_csv(DATA_DIR+'dseb63_application_train.csv', **read_params)
test = pd.read_csv(DATA_DIR+'dseb63_application_test.csv', **read_params)

labels = train['TARGET'].values
del train['TARGET']
sk_id_curr = np.hstack([
    train['SK_ID_CURR'].values,
    test['SK_ID_CURR'].values
])
del train['SK_ID_CURR'], test['SK_ID_CURR']

np.save('labels.npy', labels)
np.save('sk_id_curr.npy', sk_id_curr)

data = pd.concat([train, test], axis=0, ignore_index=True)

feather.write_dataframe(data, 'data.ftr')
