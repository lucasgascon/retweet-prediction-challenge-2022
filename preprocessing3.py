#%%

import pandas as pd
from verstack.stratified_continuous_split import scsplit # pip install verstack
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from nlp import create_vectorizer_text, preprocess_text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_data
from gensim.models import Word2Vec
import numpy as np

def create_vectorizer_text():
    train_data = pd.read_csv("train.csv")
    train_data = train_data.drop(columns='retweets_count')
    evaluation_data = pd.read_csv("evaluation.csv")
    X = pd.concat([train_data, evaluation_data], axis = 0)
    text_list = X['text'].apply(lambda x : x.split(' '))
    vectorizer_text1 = Word2Vec(vector_size=20, window=7, min_count=1, workers=-1)
    vectorizer_text1.build_vocab(text_list)
    vectorizer_text1.train(text_list, total_examples = vectorizer_text1.corpus_count, epochs = 30)
    return vectorizer_text1

def preprocess_text(X, vectorizer_text1):
    text = X['text'].apply(lambda x : x.split(' '))
    X_pretext = []
    for tweet in text:
        moy = np.array([0.0 for i in range(50)])
        for word in tweet:
            moy+=vectorizer_text1.wv[word]
        X_pretext.append(moy/len(tweet))
    X_clean = []
    for ligne in X_pretext:
        X_clean.append(list(ligne)) 
    columns = [str(i) for i in range(len(X_clean[0]))]
    df_clean = pd.DataFrame(columns=columns, index=X.index)
    for i, j in enumerate(X.index):
        df_clean.loc[j] = X_clean[i]
    X = pd.concat([X, df_clean], axis = 1)
    return X


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
        vectorizer_hashtags = TfidfVectorizer(max_features=5, stop_words=stopwords.words('french'))
        hashtags_text = pd.DataFrame(vectorizer_hashtags.fit_transform(hashtags_text).toarray(), index = hashtags.index)
    else : 
        hashtags_text = pd.DataFrame(vectorizer_hashtags.transform(hashtags_text).toarray(), index = hashtags.index)
    X = pd.concat([X, hashtags_text], axis = 1)
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


def add_variables(X, train, vectorizer_text = None, vectorizer_hashtags = None):
    X["len_text"] = X["text"].apply(lambda x: len(x))
    X = preprocess_text(X, vectorizer_text)
    X = preprocess_time(X)
    X, vectorizer_hashtags = preprocess_hashtags(X, train, vectorizer_hashtags)
    X = preprocess_urls(X)
    X = add_sentiments(X)
    return X, vectorizer_text, vectorizer_hashtags

def select_columns(X):
    X = X.drop(['text','mentions','urls','hashtags', 'timestamp','TweetID'], axis = 1)
    return X


def preprocessing(X, train, vectorizer_text = None, vectorizer_hashtags = None):
    X, vectorizer_text, vectorizer_hashtags = add_variables(X, train, vectorizer_text, vectorizer_hashtags)
    X = select_columns(X)
    return X, vectorizer_text, vectorizer_hashtags

def load_train_data(test, vectorizer_text):
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
    X_train, vectorizer_text, vectorizer_hashtags = preprocessing(X_train, train = True, vectorizer_text = vectorizer_text)
    if test == True: 
        X_test, vectorizer_text, vectorizer_hashtags = preprocessing(
                                                                        X_test, 
                                                                        train = False, 
                                                                        vectorizer_text = vectorizer_text,
                                                                        vectorizer_hashtags = vectorizer_hashtags, 
                                                                        )
        return X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags

    else: return X_train, y_train, vectorizer_text, vectorizer_hashtags


def load_validation_data(vectorizer_text, vectorizer_hashtags):
    eval_data = pd.read_csv("evaluation.csv")
    X_eval, vectorizer_text, vectorizer_hashtags = preprocessing(eval_data, 
                        train=False, 
                        vectorizer_text= vectorizer_text,
                        vectorizer_hashtags=vectorizer_hashtags,
                        )
    return X_eval


vectorizer_text1 = create_vectorizer_text()
X_train, y_train, X_test, y_test, vectorizer_text1, vectorizer_hashtags = load_train_data(test=True, vectorizer_text = vectorizer_text1)
X, y, vectorizer_text1, vectorizer_hashtags = load_train_data (test=False, vectorizer_text = vectorizer_text1)
X_val = load_validation_data(
    vectorizer_text=vectorizer_text1,
    vectorizer_hashtags=vectorizer_hashtags,
    )
save_data('preprocess_data', X, y, X_train, y_train, X_test, y_test, X_val)
# %%