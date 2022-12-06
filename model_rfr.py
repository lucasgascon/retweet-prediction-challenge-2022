#%%
from sklearn.ensemble import RandomForestRegressor
import time
import pickle

def train_custom_model(X_train, y_train):
    reg = RandomForestRegressor(
        max_depth = 18,
        n_estimators = 100,
        # criterion = 'absolute_error',
        criterion = 'squared_error',
        max_features = 'auto',
        random_state = 0,
        n_jobs = 5,
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

