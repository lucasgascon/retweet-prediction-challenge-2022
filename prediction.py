#%%

import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack

from nltk.corpus import stopwords

import os

###################################
# Once we finalized our features and model we can train it using the whole training set and then produce prediction for the evaluating dataset
###################################
# Load the evaluation data
eval_data = pd.read_csv("evaluation.csv")

from preprocessing import load_train_data, load_validation_data
X_train, y_train, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data(test=False)
np.save('data/array/X', X_train)
np.save('data/array/y', y_train.to_numpy())

X_train = np.load('data/array/X.npy')
y_train = np.load('data/array/y.npy')

# We fit our model using the training data
#%%
from model import train_custom_model
# reg = RandomForestRegressor()
reg = train_custom_model(X_train, y_train)
reg.fit(X_train, y_train)

X_val = load_validation_data(
    vectorizer_text=vectorizer_text,
    vectorizer_hashtags=vectorizer_hashtags, 
    std_clf= std_clf,
    )
# X_val.to_csv('data/X_val')
# X_val = pd.read_csv('data/X_val')
#%%

# Predict the number of retweets for the evaluation dataset
y_pred = reg.predict(X_val)
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
