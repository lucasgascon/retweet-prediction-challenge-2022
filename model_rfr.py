#%%
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import csv
import os
from utils import load_data

name_folder = 'old_csv'
X, y, X_train, y_train, X_test, y_test, X_val = load_data(name_folder)

def custom_model(X_train, y_train, X_test, save= False):
    start_time = time.time()
    rfc = RandomForestClassifier(n_jobs = 6)
    y_train_2 = y_train['retweets_count'].apply(lambda x : 1 if (x>0) else 0)
    rfc.fit(X_train, y_train_2)
    classifier_test = rfc.predict(X_test)
    
    # save randomforest classifier
    if save:
        filename = './model/randomforest_classifier_1.sav'
        pickle.dump(rfc, open(filename, 'wb'))

    # reg = RandomForestRegressor()
    reg = LGBMRegressor(n_jobs = 6)
    
    reg.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

    # save randomforest regressor
    if save:
        filename = './model/lgbmr_1.sav'
        pickle.dump(reg, open(filename, 'wb'))
    
    y_pred = reg.predict(X_test)
    y_pred = [y_pred[i]*classifier_test[i] for i in range(len(y_pred))]
    y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

    return y_pred

def rfr(X_train, y_train, X_test, save= False):
    start_time = time.time()
    

    reg = RandomForestRegressor(n_jobs = 6)
    # reg = LGBMRegressor(n_jobs = 6)
    
    reg.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

    # save randomforest regressor
    if save:
        filename = './model/rfc_1.sav'
        pickle.dump(reg, open(filename, 'wb'))
    
    y_pred = reg.predict(X_test)
    y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

    return y_pred


y_pred = rfr(X_train, y_train, X_test, save = True)
y_pred = custom_model(X_train, y_train, X_test, save = True)
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))


