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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nltk.corpus import stopwords

from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm 

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

import torch
from torch.utils.data import TensorDataset

seed = 12
#%%

# Load the training data
train_data = pd.read_csv("train.csv")
y = train_data['retweets_count']

X = train_data.drop(['retweets_count'], axis=1)

#%%


def preprocess_text(X, train = True, vectorizer_text = None):
    if train == True:
        vectorizer_text = TfidfVectorizer(max_features=170, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer_text.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer_text.transform(X['text']).toarray(), index = X.index)
    X = pd.concat([X, X_text], axis = 1)
    return X, vectorizer_text

def preprocess_time(X):
    X['month'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%m")))
    X['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))
    return X

def preprocess_hashtags(X, train = True, vectorizer_hashtags = None):
    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags_count = hashtags.apply(lambda x : len(x))
    X['hashtags_count'] = hashtags_count

    hashtags_text = hashtags.apply(lambda x : ' '.join(x))
    if train : 
        vectorizer_hashtags = TfidfVectorizer(max_features=10, stop_words=stopwords.words('french'))
        hashtags_text = pd.DataFrame(vectorizer_hashtags.fit_transform(hashtags_text).toarray(), index = hashtags.index)
    else : 
        hashtags_text = pd.DataFrame(vectorizer_hashtags.transform(hashtags_text).toarray(), index = hashtags.index)
    
    X = pd.concat([X, hashtags_text], axis = 1)
    return X, vectorizer_hashtags

def preprocess_urls(X):
    urls = X['urls'].apply(lambda x : x[2:-2].split(','))
    urls.apply(lambda x : x.remove('') if '' in x else x)
    urls = urls.apply(lambda x : len(x))
    X['urls_count'] = urls
    return X

def add_sentiments(X):
    sia = SentimentIntensityAnalyzer()
    X['polarity_scores_neg'] = X['text'].apply(lambda text : sia.polarity_scores(text)['neg'])
    X['polarity_scores_pos'] = X['text'].apply(lambda text : sia.polarity_scores(text)['pos'])
    X['polarity_scores_compound'] = X['text'].apply(lambda text : sia.polarity_scores(text)['compound'])
    return X


def add_variables(X, train, vectorizer_text = None, vectorizer_hashtags = None):
    X, vectorizer_text = preprocess_text(X, train, vectorizer_text)
    X = preprocess_time(X)
    X, vectorizer_hashtags = preprocess_hashtags(X, train, vectorizer_hashtags)
    X = preprocess_urls(X)

    X = add_sentiments(X)

    return X, vectorizer_text, vectorizer_hashtags

def select_columns(X):
    X = X.drop(['text','mentions','urls','hashtags', 'timestamp','TweetID'], axis =1)
    return X

def pipeline(X, train, std_clf = None):
    if train:
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=50))
        std_clf = make_pipeline(StandardScaler())
        std_clf.fit(X)
        X_transformed = std_clf.transform(X)
        
    else:
        X_transformed = std_clf.transform(X)
    # return X_transformed, std_clf
    return X, std_clf

def preprocessing(X, train, vectorizer_text = None, vectorizer_hashtags = None, std_clf = None):
    X, vectorizer_text, vectorizer_hashtags = add_variables(X, train, vectorizer_text, vectorizer_hashtags)
    X = select_columns(X)
    X_transformed, std_clf = pipeline(X, train, std_clf)
    return X_transformed, vectorizer_text, vectorizer_hashtags, std_clf

#%%
X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)
X_train, vectorizer_text, vectorizer_hashtags = add_variables(X_train, True)
X_test, vectorizer_text, vectorizer_hashtags = add_variables(X_test, False, 
                    vectorizer_text = vectorizer_text, 
                    vectorizer_hashtags = vectorizer_hashtags )

#%%


# X_train, vectorizer_text, vectorizer_hashtags, std_clf = preprocessing(X_train, train = True)
# X_test, vectorizer_text, vectorizer_hashtags, std_clf  = preprocessing(X_test, 
#                     train = False, 
#                     vectorizer_text = vectorizer_text, 
#                     vectorizer_hashtags = vectorizer_hashtags, 
#                     std_clf = std_clf,
#                     )
#%%
def load_train_data(test = True):
    # Load the training data
    train_data = pd.read_csv("train.csv")

    if test == True:
        # Here we split our training data into training and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
        # scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
        X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)
        X_test = X_test.drop(['retweets_count'], axis=1)
        X_train = X_train.drop(['retweets_count'], axis=1)
    else :
        y_train = train_data['retweets_count']
        X_train = train_data.drop(['retweets_count'], axis=1)
    
    # We preprocess the data
    X_train, vectorizer_text, vectorizer_hashtags, std_clf = preprocessing(X_train, train = True)
    if test == True:
        X_test, vectorizer_text, vectorizer_hashtags, std_clf  = preprocessing(X_test, 
                    train = False, 
                    vectorizer_text = vectorizer_text, 
                    vectorizer_hashtags = vectorizer_hashtags, 
                    std_clf = std_clf,
                    )
        return X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags, std_clf

    else: return X_train, y_train, vectorizer_text, vectorizer_hashtags, std_clf

def load_validation_data(vectorizer_text, vectorizer_hashtags, std_clf):
    eval_data = pd.read_csv("evaluation.csv")
    X_eval, vectorizer_text, vectorizer_hashtags, std_clf = preprocessing(eval_data, 
                        train=False, 
                        vectorizer_text= vectorizer_text,
                        vectorizer_hashtags=vectorizer_hashtags,
                        std_clf = std_clf,
                        )
    return X_eval

#%%

X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data(test=True)

X_train

# %%
