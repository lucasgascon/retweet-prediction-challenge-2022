# %%
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
import pandas as pd
# from model_rfr import train_custom_model



dir = 'csv'
X_train = pd.read_csv('data/' + dir + '/X_train.csv')
X_test = pd.read_csv('data/' + dir + '/X_test.csv')
y_train = pd.read_csv('data/' + dir + '/y_train.csv')
y_test = pd.read_csv('data/' + dir + '/y_test.csv')


# %%
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

y_train_bracket2 = np.array(y_train_bracket2['retweets_count'])
y_test_bracket2 = np.array(y_test_bracket2['retweets_count'])

y_train_bracket1 = np.array(y_train_bracket1['retweets_count'])
y_test_bracket1 = np.array(y_test_bracket1['retweets_count'])

y_train_bracket3 = np.array(y_train_bracket3['retweets_count'])
y_test_bracket3 = np.array(y_test_bracket3['retweets_count'])


#%%
# X_train = pd.read_csv('data/csv/X_train.csv', index_col = 0)
# X_test = pd.read_csv('data/csv/X_test.csv', index_col = 0)
# y_train = pd.read_csv('data/csv/y_train.csv', index_col=0)
# y_test = pd.read_csv('data/csv/y_test.csv', index_col=0)
# %%
# reg = train_custom_model(X_train, y_train)

dt_reg = DecisionTreeRegressor(random_state=0, criterion='friedman_mse', ccp_alpha=0.1)
dt_reg.fit(X_train_bracket2, y_train_bracket2)
y_pred_dt = dt_reg.predict(X_test_bracket2)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred_dt]
print("Prediction error:", mean_absolute_error(y_true=y_test_bracket2, y_pred=y_pred))

# %%
dt_reg3 = DecisionTreeRegressor(random_state=0, criterion='friedman_mse', ccp_alpha=0.1)
dt_reg3.fit(X_train_bracket3, y_train_bracket3)
y_pred_dt = dt_reg3.predict(X_test_bracket3)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred_dt]
print("Prediction error:", mean_absolute_error(y_true=y_test_bracket3, y_pred=y_pred))
   
dt_reg1 = DecisionTreeRegressor(random_state=0, criterion='friedman_mse', ccp_alpha=0.1)
dt_reg1.fit(X_train_bracket1, y_train_bracket1)
y_pred_dt = dt_reg1.predict(X_test_bracket1)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred_dt]
print("Prediction error:", mean_absolute_error(y_true=y_test_bracket1, y_pred=y_pred))
   
# %%
y_pred_dt_1 = dt_reg1.predict(X_test_bracket1)
y_pred_1 = [int(value) if value >= 0 else 0 for value in y_pred_dt_1]
y_pred_dt_2 = dt_reg.predict(X_test_bracket2)
y_pred_2 = [int(value) if value >= 0 else 0 for value in y_pred_dt_2]
y_pred_dt_3 = dt_reg3.predict(X_test_bracket3)
y_pred_3 = [int(value) if value >= 0 else 0 for value in y_pred_dt_3]

# %%

y_pred = np.concatenate((y_pred_dt_1, y_pred_dt_2, y_pred_dt_3), axis=None)
y_test = np.concatenate((y_test_bracket1, y_test_bracket2, y_test_bracket3), axis=None)
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
 
   
   
    # elif method == "lasso":
    #     clf = Lasso(alpha=0.2)
    #     clf.fit(X_train,y_train)
    #     y_pred_clf = clf.predict(X_test)
    #     y_pred = [int(value) if value >= 0 else 0 for value in y_pred_clf]
    #     print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
    #     return
    # elif method =='rfr':
    #     reg = RandomForestRegressor()
    #     reg.fit(X_train, y_train)
    #     y_pred = reg.predict(X_test)
    #     y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
    #     print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
    # # elif method=='tcm':
    # #     y_pred_dt = reg = train_custom_model(X_train, y_train, X_test)
    #     print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
    #     return

# %%


# feature_importance = pd.Series(reg.feature_importances_, index= X_train.columns)
# feature_importance.sort_values(ascending=False)
#%%
y_train_bracket1

# from model import train_nnrf, get_normal_counter
# regr, reg = train_nnrf(X_train, y_train)
# nn_y_test_predict = regr.predict(np.log(X_test + 1))
# rf_test_predict = reg.predict(X_test)
# y_pred = get_normal_counter(nn_y_test_predict + rf_test_predict, logarithm="e")
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))



# %%
