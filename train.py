# %%
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn import svm
from utils import load_data
from model_rfr import custom_model

# GradientBoostingRegressor MAE error: 6.55
X, y, X_train, y_train, X_test, y_test, X_val = load_data('old_csv') 

# MAE error:
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv150')

# MAE error:
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv50')

# reg = train_custom_model(X_train, y_train)
reg = svm.SVR()
# reg = KNeighborsRegressor(n_jobs = -1)
# reg = GradientBoostingRegressor (loss = 'huber')
# reg = RandomForestRegressor(n_jobs = 6)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
# %%
