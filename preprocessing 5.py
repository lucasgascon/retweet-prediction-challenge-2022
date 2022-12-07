#%%

import pandas as pd
from verstack.stratified_continuous_split import scsplit # pip install verstack
from sklearn.preprocessing import StandardScaler
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_data, save_data_numpy
import numpy as np
import os

def preprocess_text(X, train = True, vectorizer_text = None):
    if train == True:
        vectorizer_text = TfidfVectorizer(max_features=150, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer_text.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer_text.transform(X['text']).toarray(), index = X.index)
    X = pd.concat([X, X_text], axis = 1)
    return X, vectorizer_text

def preprocess_time(X):
    X['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))
    X['day'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%a"))
    day = {'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri': 5, 'Sat':6, 'Sun':7}
    X['day'] = X['day'].apply(lambda x : day[x])
    X['am_pm'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%p"))
    am_pm = {'AM': 0, 'PM':1}
    X['am_pm'] = X['am_pm'].apply(lambda x : am_pm[x])
    return X

def preprocess_hashtags(X):
    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags_count = hashtags.apply(lambda x : len(x))
    X['hashtag_count'] = hashtags_count
    return X

def preprocess_urls(X):
    urls = X['urls'].apply(lambda x : x[2:-2].split(','))
    urls.apply(lambda x : x.remove('') if '' in x else x)
    urls = urls.apply(lambda x : len(x))
    X['url_count'] = urls
    return X

def add_sentiments(X):
    sia = SentimentIntensityAnalyzer()
    X['polarity_scores_neg'] = X['text'].apply(lambda text : sia.polarity_scores(text)['neg'])
    X['polarity_scores_pos'] = X['text'].apply(lambda text : sia.polarity_scores(text)['pos'])
    X['polarity_scores_compound'] = X['text'].apply(lambda text : sia.polarity_scores(text)['compound'])
    return X

def add_variables(X, train, vectorizer_text = None):
    X["len_text"] = X["text"].apply(lambda x: len(x))

    X, vectorizer_text = preprocess_text(X, train, vectorizer_text)
    X = preprocess_time(X)
    X = preprocess_hashtags(X)
    X = preprocess_urls(X)
    X = add_sentiments(X)
    return X, vectorizer_text

def select_columns(X):
    X = X.drop(['text','mentions','urls','hashtags', 'timestamp','TweetID'], axis = 1)
    return X

def pipeline(X, train, std_clf = None):
    if train:
        std_clf = make_pipeline(StandardScaler())
        std_clf.fit(X)
        X_transformed = std_clf.transform(X)
    else:
        X_transformed = std_clf.transform(X)
    return X_transformed, std_clf

def preprocessing(X, train, vectorizer_text = None, std_clf = None):
    X, vectorizer_text = add_variables(X, train, vectorizer_text)
    X = select_columns(X)
    X_transformed, std_clf = pipeline(X, train, std_clf)
    return X_transformed, vectorizer_text, std_clf

def load_train_data(test):
    # Load the training data
    train_data = pd.read_csv("train.csv")

    if test == True:
        X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)
        X_test = X_test.drop(['retweets_count'], axis=1)
        X_train = X_train.drop(['retweets_count'], axis=1)
    else :
        y_train = train_data['retweets_count']
        X_train = train_data.drop(['retweets_count'], axis=1)
    
    # We preprocess the data
    X_train, vectorizer_text, std_clf = preprocessing(X_train, train = True)
    if test == True: 
        X_test, vectorizer_text, std_clf  = preprocessing(
                                X_test, 
                                train = False, 
                                vectorizer_text = vectorizer_text,
                                std_clf = std_clf,
                                )
        return X_train, y_train, X_test, y_test, vectorizer_text, std_clf

    else: return X_train, y_train, vectorizer_text, std_clf

def load_validation_data(vectorizer_text, std_clf):
    eval_data = pd.read_csv("evaluation.csv")
    X_eval, vectorizer_text, std_clf = preprocessing(eval_data, 
                        train=False, 
                        vectorizer_text= vectorizer_text,
                        std_clf = std_clf,
                        )
    return X_eval

X_train, y_train, X_test, y_test, vectorizer_text, std_clf = load_train_data(test=True)
X, y, vectorizer_text, std_clf = load_train_data (test=False)
X_val = load_validation_data(
    vectorizer_text=vectorizer_text,
    std_clf = std_clf,
    )

save_data_numpy('preprocessing_stscaler', X, y, X_train, y_train, X_test, y_test, X_val)


# %%
