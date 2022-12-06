# %%

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import time
import pandas as pd
import torch

# X_train = pd.read_csv('data_legacy/csv/X_train.csv', index_col=0)
# X_test = pd.read_csv('data_legacy/csv/X_test.csv', index_col=0)
# y_train = pd.read_csv('data_legacy/csv/y_train.csv', index_col=0)
# y_test = pd.read_csv('data_legacy/csv/y_test.csv', index_col=0)


X_train = pd.read_csv('data2/csv/X_train.csv', index_col=0)
X_test = pd.read_csv('data2/csv/X_test.csv', index_col=0)
y_train = pd.read_csv('data2/csv/y_train.csv', index_col=0)
y_test = pd.read_csv('data2/csv/y_test.csv', index_col=0)

# dir = 'scale'
# X_train = np.load('data/' + dir + '/X_train.npy')
# X_test = np.load('data/' + dir + '/X_test.npy')
# y_train = np.load('data/' + dir + '/y_train.npy')
# y_test = np.load('data/' + dir + '/y_test.npy')


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
# Set parameters
xgb_params = {
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 1,
    'booster' : 'gbtree',
    'objective': 'reg:squarederror',
    'lambda': 1.2,   
    'alpha': 0.4, 
    'n_jobs': 5,
}


num_round = 100
start_time = time.time()
bst = xgb.train(xgb_params, dtrain, num_round)
elapsed_time = time.time() - start_time
print("took {} seconds for fitting".format(elapsed_time))

y_pred = bst.predict(dtest)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))


# xgb1 = XGBRegressor()

# parameters = {'nthread':[4],
#               'objective':['reg:squarederror'],
#               'booster' : ['gbtree'],
#               'learning_rate': [0.05, 0.10, 0.15],
#               'max_depth': [6],
#               'min_child_weight': [4],
#               'subsample': [0.8, 1],
#               'colsample_bytree': [0.7],
#               'n_estimators': [100],
#               'eta': [0.01],
#               'lambda': [0.4],
#               'alpha': [0.4],
#               }

# xgb_grid = GridSearchCV(xgb1,
#                         parameters,
#                         cv = 2,
#                         n_jobs = 2,
#                         verbose=True)

# xgb_grid.fit(X_train,y_train)

# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)

# grid_predictions = xgb_grid.predict(X_test) 
# y_pred = [int(value) if value >= 0 else 0 for value in grid_predictions]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pre# %%
