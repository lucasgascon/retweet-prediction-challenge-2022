#%%

import csv
import pandas as pd
import os
from utils import load_data
from model_rfr import custom_model, rfr

###################################
# Once we finalized our features and model we can train it using the whole training set and then produce prediction for the evaluating dataset
###################################
def main_prediction():

    eval_data = pd.read_csv("evaluation.csv")

    X, y, X_train, y_train, X_test, y_test, X_val = load_data('preprocess_data')
    # X, y, X_train, y_train, X_test, y_test, X_val = load_data('old_csv')

    y_pred = custom_model(X, y, X_val, save = False)
    # y_pred = rfr(X, y, X_val, save = False)

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
    
# main_prediction()
# %%
