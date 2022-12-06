#%%

import csv
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import os
from utils import load_data
from model import train_custom_model, rfr

###################################
# Once we finalized our features and model we can train it using the whole training set and then produce prediction for the evaluating dataset
###################################
eval_data = pd.read_csv("evaluation.csv")

X, y, X_train, y_train, X_test, y_test, X_val = load_data('preprocess_data')

y_pred = rfr(X, y, X_val)

# Dump the results into a file that follows the required Kaggle template
with open("gbr_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "retweets_count"])
    for index, prediction in enumerate(y_pred):
        writer.writerow([str(eval_data['TweetID'].iloc[index]) , str(int(prediction))])

os.makedirs('pred', exist_ok=True)  
pred = pd.read_csv('gbr_predictions.txt')
pred.set_index('TweetID', inplace= True)
pred.to_csv('pred/prediction.csv')