# %%
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
import pandas as pd
from model_rfr import custom_model
from utils import load_data
import time
from catboost import CatBoostRegressor
import pickle

dir = 'csv'
X, y, X_train, y_train, X_test, y_test, X_val = load_data('preprocessing')

def model(X_train, y_train, X_test, save= False):
    start_time = time.time()

    reg = CatBoostRegressor(
        iterations = 10000,
        random_state = 42,
        )
    
    reg.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

    # save randomforest regressor
    if save:
        filename = 'model/rfc_1.sav'
        pickle.dump(reg, open(filename, 'wb'))
    
    y_pred = reg.predict(X_test)
    y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
    

    return y_pred


# %%

### Separating the data in three brackets
X_train_bracket1 = X_train[X_train['verified']==1]
y_train_bracket1 = y_train[X_train['verified']==1]

X_train_bracket2 = X_train[(X_train['verified']==0) & (X_train['favorites_count']<100)]
y_train_bracket2 = y_train[(X_train['verified']==0) & (X_train['favorites_count']<100)]

X_train_bracket3 = X_train[(X_train['verified']==0) & (X_train['favorites_count']>=100)]
y_train_bracket3 = y_train[(X_train['verified']==0) & (X_train['favorites_count']>=100)]


X_test_bracket1 = X_test[X_test['verified']==1]
y_test_bracket1 = y_test[X_test['verified']==1]

X_test_bracket2 = X_test[(X_test['verified']==0) & (X_test['favorites_count']<100)]
y_test_bracket2 = y_test[(X_test['verified']==0) & (X_test['favorites_count']<100)]

X_test_bracket3 = X_test[(X_test['verified']==0) & (X_test['favorites_count']>=100)]
y_test_bracket3 = y_test[(X_test['verified']==0) & (X_test['favorites_count']>=100)]



# %%

# predicting for each brackets the number of retweets
y_pred_dt_2 = model(X_train_bracket2, y_train_bracket2, X_test_bracket2)
print("Prediction error for bracket 2:", mean_absolute_error(y_true=y_test_bracket2, y_pred=y_pred_dt_2))

y_pred_dt_3 = model(X_train_bracket3, y_train_bracket3, X_test_bracket3)
print("Prediction error for bracket 3:", mean_absolute_error(y_true=y_test_bracket3, y_pred=y_pred_dt_3))
   
y_pred_dt_1 = model(X_train_bracket1, y_train_bracket1, X_test_bracket1)
print("Prediction error for bracket 1:", mean_absolute_error(y_true=y_test_bracket1, y_pred=y_pred_dt_1))
   
# %%
# predicting the total results
y_pred = np.concatenate((y_pred_dt_1, y_pred_dt_2, y_pred_dt_3), axis=None)
y_test = np.concatenate((y_test_bracket1, y_test_bracket2, y_test_bracket3), axis=None)
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
 
