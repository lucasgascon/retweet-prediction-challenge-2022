#%%
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time
import pickle
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from utils import load_data, load_data_numpy

# MAE error: 6.49
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('preprocessing')

# MAE error: 6.49
# X, y, X_train, y_train, X_test, y_test, X_val = load_data_numpy('preprocessing_stscaler')

# MAE error: 6.09
X, y, X_train, y_train, X_test, y_test, X_val = load_data('old_csv') 

# MAE error: 6.96
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv150')

# MAE error: 7.46
# X, y, X_train, y_train, X_test, y_test, X_val = load_data('csv50')

def custom_model(X_train, y_train, X_test, save= False):
    start_time = time.time()

    rfc = RandomForestClassifier(
        n_estimators = 300,
        criterion = 'entropy',
        max_depth = 10,
        n_jobs = -1,
        random_state = 42,
        verbose = 5,
        )

    y_train_2 = y_train['retweets_count'].apply(lambda x : 1 if (x>0) else 0)
    rfc.fit(X_train, y_train_2)
    classifier_test = rfc.predict(X_test)
    
    # save randomforest classifier
    if save:
        filename = './model/randomforest_classifier_1.sav'
        pickle.dump(rfc, open(filename, 'wb'))

    reg = LGBMRegressor(
        boosting_type='gbdt',
        learning_rate = 0.037,
        n_estimators = 300,
        n_jobs = -1,
        random_state = 42, 
        verbose = 5,
    )
    
    reg.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print("took {} seconds for fitting".format(elapsed_time))

    # save LGBM regressor
    if save:
        filename = './model/lgbmr_1.sav'
        pickle.dump(reg, open(filename, 'wb'))
    
    y_pred = reg.predict(X_test)
    y_pred = [y_pred[i]*classifier_test[i] for i in range(len(y_pred))]
    y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

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

y_pred = rfr(X_train, y_train, X_test, save = False)
# y_pred = custom_model(X_train, y_train, X_test, save = False)
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))




# %%
