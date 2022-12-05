#%%

import pandas as pd
from verstack.stratified_continuous_split import scsplit # pip install verstack
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from sklearn.pipeline import make_pipeline
from nlp import preprocess_text, preprocess_text_2
from test_nlp import *

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

import os

seed = 12

# Load the training data
train_data = pd.read_csv("train.csv")
y = train_data['retweets_count']
X = train_data.drop(['retweets_count'], axis=1)

#%%

def preprocess_time(X):
    X['month'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%m")))
    X['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))
    X['day'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%a"))
    day = {'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri': 5, 'Sat':6, 'Sun':7}
    X['day'] = X['day'].apply(lambda x : day[x])
    X['am_pm'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%p"))
    am_pm = {'AM': 0, 'PM':1}
    X['am_pm'] = X['am_pm'].apply(lambda x : am_pm[x])
    return X


def preprocess_hashtags(X, train = True, vectorizer_hashtags = None):
    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags_count = hashtags.apply(lambda x : len(x))
    X['hashtag_count'] = hashtags_count
    hashtags_text = hashtags.apply(lambda x : ' '.join(x))
    if train : 
        vectorizer_hashtags = TfidfVectorizer(max_features=50, stop_words=stopwords.words('french'))
        hashtags_text = pd.DataFrame(vectorizer_hashtags.fit_transform(hashtags_text).toarray(), index = hashtags.index)
    else : 
        hashtags_text = pd.DataFrame(vectorizer_hashtags.transform(hashtags_text).toarray(), index = hashtags.index)
    # X = pd.concat([X, hashtags_text], axis = 1)
    return X, vectorizer_hashtags

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

def other_variables(X_train):

    mentions = X_train['hashtags'].apply(lambda x : x[2:-2].split(','))
    mentions.apply(lambda x : x.remove('') if '' in x else x)
    mentions_count = mentions.apply(lambda x : len(x))
    X_train['mention_count'] = mentions_count
    
    X_train["len_text"] = X_train["text"].apply(lambda x: len(x))
    
    # approx length of tweets = sum of all h/e/m/url
    X_train["tlen"] = X_train["len_text"] + X_train["hashtag_count"] + X_train["mention_count"] + X_train[
        "url_count"]

    return X_train

def add_variables(X, train, vectorizer_text = None, vectorizer_hashtags = None):
    
    # X, vectorizer_text = preprocess_text(X, train, vectorizer_text)
    X, vectorizer_text = preprocess_text_2(X, train, vectorizer_text)
    X, vectorizer_text = create_train_df(X, train = True, vectorizer_text = None, modele = 'base', function = preprocess_text_6)

    X = preprocess_time(X)
    X, vectorizer_hashtags = preprocess_hashtags(X, train, vectorizer_hashtags)
    X = preprocess_urls(X)
    X = add_sentiments(X)
    X = other_variables(X)
    return X, vectorizer_text, vectorizer_hashtags

def select_columns(X):
    X = X.drop(['text','mentions','urls','hashtags', 'timestamp','TweetID'], axis = 1)
    return X

def pipeline(X, train, std_clf = None):
    if train:
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=50))
        # std_clf = make_pipeline(StandardScaler())
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

# dir = 'scale'
# X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data(test=True)
# np.save('data/' + dir + '/X_train', X_train)
# np.save('data/' + dir + '/X_test', X_test)
# np.save('data/' + dir + '/y_train', y_train.to_numpy())
# np.save('data/' + dir + '/y_test', y_test.to_numpy())
# X, y, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data (test=False)
# np.save('data/' + dir + '/X', X)
# np.save('data/' + dir + '/y', y.to_numpy())
# X_val = load_validation_data(
#     vectorizer_text=vectorizer_text,
#     vectorizer_hashtags=vectorizer_hashtags,
#     std_clf = std_clf,
#     )
# np.save('data/' + dir + '/X_val', X_train)


X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data(test=True)

X_train.head()

X, y, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data (test=False)
X_val = load_validation_data(
    vectorizer_text=vectorizer_text,
    vectorizer_hashtags=vectorizer_hashtags,
    std_clf = std_clf,
    )

os.makedirs('data6', exist_ok=True)  
X_train.to_csv('data6/csv/X_train.csv')
X_test.to_csv('data6/csv/X_test.csv')
X_val.to_csv('data6/csv/X_val.csv')
X.to_csv('data6/csv/X.csv')
y_train.to_csv('data6/csv/y_train.csv')
y_test.to_csv('data6/csv/y_test.csv')
y.to_csv('data6/csv/y.csv')







# %%
