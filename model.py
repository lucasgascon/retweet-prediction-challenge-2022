#%%
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import numpy as np
import pandas as pd


def train_custom_model(X_train, y_train):
    reg = RandomForestRegressor(
        max_depth = 18,
        n_estimators = 100,
        criterion = 'squared_error',
        max_features = 'auto',
        random_state = 0,
        n_jobs = -1,
        verbose = 5,
    )
    start_time = time.time()
    reg.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    # save randomforest regressor
    filename = './model/randomforest_regressor_mine3.sav'
    pickle.dump(reg, open(filename, 'wb'))
    return reg


def get_normal_counter(n, logarithm="10"):
    """ Get normal counter x from log(x+1) value """
    if logarithm == "10":
        return np.array([max(0, x) for x in (np.power(10, n).round().astype(int)-1)])
    else:
        return np.array([max(0, x) for x in (np.exp(n).round().astype(int)-1)])


def train_nnrf(X_train, y_train):
    """ Train and store NNRF (Neural Networks - MLP + Random Forest) """
    regr = MLPRegressor(random_state=7,
                        hidden_layer_sizes=(64, 32, 16, 8, 8),
                        batch_size=1024,
                        learning_rate_init=.01,
                        early_stopping=False,
                        verbose=True,
                        shuffle=True,
                        n_iter_no_change=10)

    y_train_logscale = np.log(y_train + 1.)

    y_train_logscale = y_train_logscale.astype(float)

    # fit
    start_time = time.time()
    regr.fit(np.log(X_train + 1.), y_train_logscale)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

    filename = './model/nnnoval_shuffle.sav'
    pickle.dump(regr, open(filename, 'wb'))

    lr_y_train_predict = regr.predict(np.log(X_train + 1.))
    # for training residual RF
    rf_y_train = y_train_logscale - lr_y_train_predict

    reg = RandomForestRegressor(max_depth=None,
                                n_estimators=500,
                                random_state=77,
                                n_jobs=-1,
                                verbose=5)
    start_time = time.time()
    reg.fit(X_train, rf_y_train, )
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    # save randomforest regressor
    filename = './model/randomforest_regressor_500e.sav'
    pickle.dump(reg, open(filename, 'wb'))

    return regr, reg


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
def train_lrrf(X_train, y_train):
    """ Train and store LRRF (LR + Random Forest) """
    y_train_logscale = np.log(y_train + 1.)
    regr = LinearRegression(fit_intercept=False).fit(np.log(X_train+1), y_train_logscale)
    # filename = './model/lr.sav'
    # pickle.dump(reg, open(filename, 'wb'))

    # predict
    lr_y_train_predict = regr.predict(np.log(X_train + 1))
    lr_yhat = get_normal_counter(lr_y_train_predict, logarithm="e")
    print(mean_squared_log_error(lr_yhat, y_train))

    # for training residual RF
    rf_y_train = y_train_logscale - lr_y_train_predict

    reg = RandomForestRegressor(max_depth=20,
                                n_estimators=500,
                                random_state=7,
                                n_jobs=-1,
                                verbose=5)
    start_time = time.time()
    reg.fit(X_train, rf_y_train, )
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    # save randomforest regressor
    # filename = './model/randomforest_regressor_500e_lrallfeatures_rs7.sav'
    # pickle.dump(reg, open(filename, 'wb'))

    return regr, reg

        