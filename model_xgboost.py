# %%

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import time
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score


# Load dataset
X_train = np.load('data/array/X_train.npy')
X_test = np.load('data/array/X_test.npy')
y_train = np.load('data/array/y_train.npy')
y_test = np.load('data/array/y_test.npy')

X = np.load('data/array/X.npy')
y = np.load('data/array/y.npy')

dtrain = xgb.DMatrix(X_train, label=y_train)

# Set parameters
xgb_params = {
    'n_estimators': 10,
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 1,
    # 'objective': 'reg:linear',
    'objective': 'reg:squarederror',
    'col_sample_bytree': 1,
    'lambda': 1.2,   
    'alpha': 0.4, 
}

# num_round = 100
# start_time = time.time()
# bst = xgb.train(xgb_params, dtrain, num_round)
# elapsed_time = time.time() - start_time
# print("took {} seconds for fitting".format(elapsed_time))

bst = xgb.XGBRegressor(
    tree_method="hist",
    n_estimators= 10,
    eta= 0.03,
    max_depth= 6,
    subsample= 1,
    # 'objective'= 'reg:linear',
    objective= 'reg:squarederror',
    )

#%%

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(bst, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores = abs(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )
#%%
# dtest = xgb.DMatrix(X_test)
# y_pred = bst.predict(dtest)
# y_pred = [round(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
#%%
# bst.fit(X_train, y_train)
# y_pred = bst.predict(X_test)
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))



# %%

# parameters = {
#     'eta': [0.01,0.04],
#     'max_depth': [2, 5, 10],
#     'lambda': [0.4, 1.2],
#     'alpha': [0.2,0.4, 0.6]}

# grid = GridSearchCV(bst, parameters)
# grid.fit(X_train,y_train)
# grid.best_estimator_.get_params()