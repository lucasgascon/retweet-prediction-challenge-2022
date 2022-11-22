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
# Load the train data 
train_data = pd.read_csv("train.csv")
# Load the evaluation data
eval_data = pd.read_csv("evaluation.csv")
# Transform our data into tfidf vectors
vectorizer = TfidfVectorizer(max_features=100, stop_words=stopwords.words('french'))
y_train = train_data['retweets_count']

X_train = train_data.drop(['retweets_count'], axis=1)

from preprocessing import preprocessing
# X_train, vectorizer, min_max_scaler = preprocessing (X_train, train = True)
X_train, vectorizer_text, vectorizer_hashtags, std_clf = preprocessing (X_train, train = True)


# We fit our model using the training data
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

X_val, vectorizer_text, vectorizer_hashtags, pca, scaler = preprocessing(eval_data, 
        train= False, 
        vectorizer_text = vectorizer_text, 
        vectorizer_hashtags = vectorizer_hashtags,
        std_clf = std_clf
        )

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
pred.to_csv('pred/out2.csv')