#%%
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
import pickle
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from utils import load_data, load_data_numpy
import numpy as np

# MAE error: 6.49
X, y, X_train, y_train, X_test, y_test, X_val = load_data('preprocessing')

# MAE error: 6.49
# X, y, X_train, y_train, X_test, y_test, X_val = load_data_numpy('preprocessing_stscaler')

# MAE error: 6.09
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('old_csv') 

# MAE error: 6.96
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv150')

# MAE error: 7.46
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv50')

def custom_model(X_train, y_train, X_test, y_test = None, save= False):
    start_time = time.time()

    rfc = RandomForestClassifier(
        n_estimators = 300,
        criterion = 'entropy',
        max_depth = 10,
        n_jobs = -1,
        random_state = 42,
        verbose = 5,
        )

    y_train_2 = y_train['retweets_count'].apply(lambda x : 2 if (x>20) else 1 if (x>7) else 0)
    rfc.fit(X_train, y_train_2)
    classifier_train = rfc.predict(X_train)
    classifier_test = rfc.predict(X_test)
    
    X_train_2 = X_train[classifier_train == 2]
    y_train_2 = y_train[classifier_train == 2]

    X_train_1 = X_train[classifier_train == 1]
    y_train_1 = y_train[classifier_train == 1]

    X_train_0 = X_train[classifier_train == 0]
    y_train_0 = y_train[classifier_train == 0]

    X_test_2 = X_test[classifier_test == 2]
    X_test_1 = X_test[classifier_test == 1]
    X_test_0 = X_test[classifier_test == 0]

    if y_test != None:
        y_test_2 = y_test[classifier_test == 2]
        y_test_1 = y_test[classifier_test == 1]
        y_test_0 = y_test[classifier_test == 0]

    reg2 = LGBMRegressor(
        boosting_type='gbdt',
        learning_rate = 0.037,
        n_estimators = 300,
        n_jobs = -1,
        random_state = 42, 
        verbose = 5,
    )
    reg2.fit(X_train_2, y_train_2)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    y_pred_2 = reg2.predict(X_test_2)

    reg1 = LGBMRegressor(
        boosting_type='gbdt',
        learning_rate = 0.037,
        n_estimators = 300,
        n_jobs = -1,
        random_state = 42, 
        verbose = 5,
    )
    reg1.fit(X_train_1, y_train_1)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    y_pred_1 = reg1.predict(X_test_1)

    reg0 = LGBMRegressor(
        boosting_type='gbdt',
        learning_rate = 0.037,
        n_estimators = 300,
        n_jobs = -1,
        random_state = 42, 
        verbose = 5,
    )
    reg0.fit(X_train_0, y_train_0)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))
    y_pred_0 = reg0.predict(X_test_0)

    y_pred = np.concatenate((y_pred_2, y_pred_1, y_pred_0), axis=None)
    y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
    if y_test != None :
        y_test = np.concatenate((y_test_2, y_test_1, y_test_0), axis=None)
        print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
    
    return y_pred

# MAE error: 6.57
# X, y, X_train, y_train, X_test, y_test, X_val = load_data_numpy('preprocessing_stscaler')

# MAE error: 6.34
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('preprocessing')

# MAE error: 6.12
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('old_csv') 

# MAE error: 7.25
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv150')

# MAE error: 7.67
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv50')

def rfr(X_train, y_train, X_test, save= False):
    start_time = time.time()

    reg = RandomForestRegressor(
        n_estimators = 100,
        criterion = 'squared_error',
        max_depth = 18,
        n_jobs = -1,
        random_state = 42,
        verbose = 5,
        )
    
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

# y_pred = rfr(X_train, y_train, X_test, save = False)
y_pred = custom_model(X_train, y_train, X_test, save = False)

#%%
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))




# %%
