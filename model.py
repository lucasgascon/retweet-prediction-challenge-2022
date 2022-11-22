# %%

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

# seed = 12

#%%
# Load the training data
train_data = pd.read_csv("train.csv")

#%%
# Here we split our training data into training and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

# We remove the actual number of retweets from our features since it is the value that we are trying to predict
X_train = X_train.drop(['retweets_count'], axis=1)
X_test = X_test.drop(['retweets_count'], axis=1)

# We preprocess the data
from preprocessing import preprocessing
X_train, vectorizer_text, vectorizer_hashtags, std_clf = preprocessing(X_train, train = True)
X_test, vectorizer_text, vectorizer_hashtags, std_clf  = preprocessing(X_test, 
            train = False, 
            vectorizer_text = vectorizer_text, 
            vectorizer_hashtags = vectorizer_hashtags, 
            std_clf = std_clf,
            )

#%%
# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
#reg = GradientBoostingRegressor()
reg = RandomForestRegressor()
#reg = LinearRegression()

#%%
# We fit our model using the training data
reg.fit(X_train, y_train)
# And then we predict the values for our testing set
y_pred = reg.predict(X_test)
# We want to make sure that all predictions are non-negative integers
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))





