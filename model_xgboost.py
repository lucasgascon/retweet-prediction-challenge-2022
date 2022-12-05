# %%

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import time
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score


dir ='array'

X_train = np.load('data/' + dir + '/X_train.npy')
X_test = np.load('data/' + dir + '/X_test.npy')
y_train = np.load('data/' + dir + '/y_train.npy')
y_test = np.load('data/' + dir + '/y_test.npy')


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Set parameters
xgb_params = {
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 1,
    # 'objective': 'reg:linear',
    'objective': 'reg:squarederror',
    'lambda': 1.2,   
    'alpha': 0.4, 
}

num_round = 100
start_time = time.time()
bst = xgb.train(xgb_params, dtrain, num_round)
elapsed_time = time.time() - start_time
print("took {} seconds for fitting".format(elapsed_time))


y_pred = bst.predict(dtest)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))

#%%

xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500],
            #   'eta': [0.01,0.04],
            #   'lambda': [0.4, 1.2],
            #   'alpha': [0.2,0.4, 0.6],
              }

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train,
         y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

model = xgb_grid.best_estimator_

y_pred = model.predict(dtest)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))

# %%
# bst2 = xgb.XGBRegressor(
#     tree_method="hist",
#     n_estimators= 10,
#     eta= 0.03,
#     max_depth= 6,
#     subsample= 1,
#     objective= 'reg:linear',
#     # objective= 'reg:squarederror',
#     )

# bst2.fit(X_train, y_train)
# y_pred = bst2.predict(X_test)
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))