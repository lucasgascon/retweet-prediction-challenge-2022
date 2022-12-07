# %%
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
import pandas as pd


dir = 'scale'
X_train = np.load('data/' + dir + '/X_train.npy')
X_test = np.load('data/' + dir + '/X_test.npy')
y_train = np.load('data/' + dir + '/y_train.npy')
y_test = np.load('data/' + dir + '/y_test.npy')


# X_train = pd.read_csv('data/csv/X_train.csv', index_col = 0)
# X_test = pd.read_csv('data/csv/X_test.csv', index_col = 0)
# y_train = pd.read_csv('data/csv/y_train.csv', index_col=0)
# y_test = pd.read_csv('data/csv/y_test.csv', index_col=0)

# %%
from model_rfr import train_custom_model
# reg = train_custom_model(X_train, y_train)


dt_reg = DecisionTreeRegressor(random_state=0, criterion='friedman_mse', ccp_alpha=0.1)
dt_reg.fit(X_train, y_train)
y_pred_dt = dt_reg.predict(X_test)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred_dt]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))


# clf = Lasso(alpha=0.2)
# clf.fit(X_train,y_train)
# y_pred_clf = clf.predict(X_test)
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred_clf]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))


# reg = RandomForestRegressor()
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))


# feature_importance = pd.Series(reg.feature_importances_, index= X_train.columns)
# feature_importance.sort_values(ascending=False)


# from model import train_nnrf, get_normal_counter
# regr, reg = train_nnrf(X_train, y_train)
# nn_y_test_predict = regr.predict(np.log(X_test + 1))
# rf_test_predict = reg.predict(X_test)
# y_pred = get_normal_counter(nn_y_test_predict + rf_test_predict, logarithm="e")
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))



# %%