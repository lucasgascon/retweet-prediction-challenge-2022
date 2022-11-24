from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import time
import pickle
import numpy as np


def train_custom_model(X_train, y_train):
    reg = RandomForestRegressor(
        max_depth = None,
        n_estimators = 1000,
        criterion = 'squared_error',
        random_state = 0,
        n_jobs = -1,
        verbose = 5,
    )
    start_time = time.time()
    reg.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    # save randomforest regressor
    filename = './model/randomforest_regressor_mine2.sav'
    pickle.dump(reg, open(filename, 'wb'))
    return reg

def train_nnrf(X_train_lr, y_train):
    """ Train and store NNRF (Neural Networks - MLP + Random Forest) """
    regr = MLPRegressor(random_state=7,
                        hidden_layer_sizes=(64, 32, 16, 8, 8),
                        batch_size=1024,
                        learning_rate_init=.01,
                        early_stopping=False,
                        verbose=True,
                        shuffle=True,
                        n_iter_no_change=10)

    #y_train_logscale = np.log(y_train + 1.)

    # fit
    start_time = time.time()
    #regr.fit(np.log(X_train_lr.values + 1), y_train_logscale)
    regr.fit(X_train_lr, y_train)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

    filename = './model/nnnoval_shuffle.sav'
    pickle.dump(regr, open(filename, 'wb'))

    #lr_y_train_predict = regr.predict(np.log(X_train_lr.values + 1))
    lr_y_train_predict = regr.predict(X_train_lr.values)
    # for training residual RF
    #rf_y_train = y_train_logscale - lr_y_train_predict
    rf_y_train = y_train - lr_y_train_predict

    reg = RandomForestRegressor(max_depth=None,
                                n_estimators=500,
                                random_state=77,
                                n_jobs=3,
                                verbose=5)
    start_time = time.time()

    #reg.fit(X_train[features].values, rf_y_train, )$
    reg.fit(X_train_lr.values, rf_y_train, )

    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    # save randomforest regressor
    filename = './model/randomforest_regressor_1000e_nnallfeatures_rs7.sav'
    pickle.dump(reg, open(filename, 'wb'))

    return regr