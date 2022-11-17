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

from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

seed = 12
#%%

# Load the training data
train_data = pd.read_csv("train.csv")
y = train_data['retweets_count']

X = train_data.drop(['retweets_count'], axis=1)

#%%

# sia = SentimentIntensityAnalyzer()

# colonne_text = list(X["text"])
# colonne_sentiment_neg = []
# colonne_sentiment_pos = []

# for text in colonne_text:
#     sent = sia.polarity_scores(text)
#     neg = sent['neg']
#     pos = sent['pos']
#     '''neu = sent['neu']
#     compound = sent['compound']'''
#     colonne_sentiment_neg.append(neg)
#     colonne_sentiment_pos.append(pos)
#%%
def preprocessing(X, train, vectorizer = None, min_max_scaler = None):
    # We set up an Tfidf Vectorizer that will use the top 100 tokens from the tweets. We also remove stopwords.
    # To do that we have to fit our training dataset and then transform both the training and testing dataset. 
    
    X_only_int = X.select_dtypes('int')

    if train == True:
        vectorizer = TfidfVectorizer(max_features=100, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer.transform(X['text']).toarray(), index = X.index)

    X_new = pd.concat([X_only_int, X_text], axis = 1)

    X_new['month'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%m")))
    X_new['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))

    X_new['mentions_count'] = X['mentions'].apply(lambda x : len(x))
    X_new['hashtags_count'] = X['hashtags'].apply(lambda x : len(x))
    X_new['urls_count'] = X['urls'].apply(lambda x : len(x))

    # Pour le moment cela baisse la qualité du modèle
    # sia = SentimentIntensityAnalyzer()
    # X_new['polarity_scores_neg'] = X['text'].apply(lambda text : sia.polarity_scores(text)['neg'])
    # X_new['polarity_scores_pos'] = X['text'].apply(lambda text : sia.polarity_scores(text)['pos'])
    # X_new['polarity_scores_compound'] = X['text'].apply(lambda text : sia.polarity_scores(text)['compound'])

    X_new = X_new.drop(['TweetID'], axis = 1)

    if train == True:
        min_max_scaler = MinMaxScaler().fit(X_new)
        X_norm = min_max_scaler.transform(X_new)
        return X_norm, vectorizer, min_max_scaler
    else: 
        X_norm = min_max_scaler.transform(X_new)
        return X_norm

# %%
