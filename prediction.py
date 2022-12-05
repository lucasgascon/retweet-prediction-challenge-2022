#%%

import csv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor
import os
import time

###################################
# Once we finalized our features and model we can train it using the whole training set and then produce prediction for the evaluating dataset
###################################
eval_data = pd.read_csv("evaluation.csv")

X_train = pd.read_csv('data7/csv/X.csv', index_col=0)
y_train = pd.read_csv('data7/csv/y.csv', index_col=0)

X_val = pd.read_csv('data7/csv/X_val.csv', index_col=0)

# We fit our model using the training data
# from model import train_custom_model
# reg = train_custom_model(X_train, y_train)

# reg = RandomForestRegressor(n_jobs = -1)
# reg.fit(X_train, y_train)
# # Predict the number of retweets for the evaluation dataset
# y_pred = reg.predict(X_val)
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# y_pred_2 = [round(value) if value >= 0 else 0 for value in y_pred]


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_val)

# Set parameters
xgb_params = {
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 1,
    'objective': 'reg:linear',
    # 'objective': 'reg:squarederror',
    'lambda': 1.2,   
    'alpha': 0.4, 
}

num_round = 100
start_time = time.time()
bst = xgb.train(xgb_params, dtrain, num_round)
elapsed_time = time.time() - start_time
print("took {} seconds for fitting".format(elapsed_time))

y_pred = bst.predict(dtest)

#%%
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

# Dump the results into a file that follows the required Kaggle template
with open("gbr_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "retweets_count"])
    for index, prediction in enumerate(y_pred):
        writer.writerow([str(eval_data['TweetID'].iloc[index]) , str(int(prediction))])

os.makedirs('pred', exist_ok=True)  
pred = pd.read_csv('gbr_predictions.txt')
pred.set_index('TweetID', inplace= True)
pred.to_csv('pred/out3.csv')
# %%
