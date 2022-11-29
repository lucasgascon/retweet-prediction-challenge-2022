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


from datetime import datetime


seed = 12
#%%

# Load the training data
train_data = pd.read_csv("train.csv")
y = train_data['retweets_count']

X = train_data.drop(['retweets_count'], axis=1)

#%%
def preprocess_hashtags(X, train = True, vectorizer = None):
    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags = hashtags.apply(lambda x : ' '.join(x))

    if train : 
        vectorizer = TfidfVectorizer(max_features=50, stop_words=stopwords.words('french'))
        hashtags = pd.DataFrame(vectorizer.fit_transform(hashtags).toarray(), index = hashtags.index)
        return hashtags, vectorizer
    else : 
        hashtags = pd.DataFrame(vectorizer.transform(hashtags).toarray(), index = hashtags.index)
        return hashtags, vectorizer
#%%
def preprocessing(X, train, vectorizer_text = None, vectorizer_hashtags = None, pca = None, scaler = None):
    # We set up an Tfidf Vectorizer that will use the top 100 tokens from the tweets. We also remove stopwords.
    # To do that we have to fit our training dataset and then transform both the training and testing dataset. 
    
    X_only_int = X.select_dtypes('int')

    if train == True:
        vectorizer_text = TfidfVectorizer(max_features=150, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer_text.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer_text.transform(X['text']).toarray(), index = X.index)

    X_new = pd.concat([X_only_int, X_text], axis = 1)

    X_new['month'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%m")))
    X_new['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))
    X_new['day'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%a"))
    day = {'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri': 5, 'Sat':6, 'Sun':7}
    X_new['day'] = X_new['day'].apply(lambda x : day[x])
    X_new['am_pm'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%p"))
    am_pm = {'AM': 0, 'PM':1}
    X_new['am_pm'] = X_new['am_pm'].apply(lambda x : am_pm[x])

    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags = hashtags.apply(lambda x : len(x))
    X_new['hashtags_count'] = hashtags

    if train == True:
        hashtags, vectorizer_hashtags = preprocess_hashtags(X, train = train)
        # X_new = pd.concat([X_new, hashtags], axis =1)
    else :
        hashtags, vectorizer_hashtags = preprocess_hashtags(X, train = train, vectorizer= vectorizer_hashtags)
        # X_new = pd.concat([X_new, hashtags], axis =1)

    urls = X['urls'].apply(lambda x : x[2:-2].split(','))
    urls.apply(lambda x : x.remove('') if '' in x else x)
    urls = urls.apply(lambda x : len(x))
    X_new['urls_count'] = urls

    # Pour le moment cela baisse la qualité du modèle
    sia = SentimentIntensityAnalyzer()
    X_new['polarity_scores_neg'] = X['text'].apply(lambda text : sia.polarity_scores(text)['neg'])
    X_new['polarity_scores_pos'] = X['text'].apply(lambda text : sia.polarity_scores(text)['pos'])
    X_new['polarity_scores_compound'] = X['text'].apply(lambda text : sia.polarity_scores(text)['compound'])

    X_new = X_new.drop(['TweetID'], axis = 1)


    if train == True:
        scaler = StandardScaler()
        X_scale = scaler.fit_transform(X_new)
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(X_scale)
        return X_new, vectorizer_text, vectorizer_hashtags, pca, scaler
    else: 
        X_scale = scaler.transform(X_new)
        X_pca = pca.transform(X_scale)
        return X_new, vectorizer_text, vectorizer_hashtags, pca, scaler

