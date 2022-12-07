#%%

import pandas as pd
#from verstack.stratified_continuous_split import scsplit # pip install verstack
from sklearn.model_selection import train_test_split
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_data
import spacy
from gensim.models import Word2Vec
import numpy as np

seed = 10


def preprocess_text(X, train = True, vector_size = 20 , window_size = 7):
    text_list = X['text'].apply(lambda x : x.split(' '))
    vectorizer_text = Word2Vec(vector_size=vector_size, window=window_size, min_count=1, workers=-1)
    vectorizer_text.build_vocab(text_list)
    vectorizer_text.train(text_list, total_examples = vectorizer_text.corpus_count, epochs = 20)
    text = X['text'].apply(lambda x : x.split(' '))
    X_pretext = []
    for tweet in text:
        moy = np.array([0.0 for i in range(vector_size)])
        for word in tweet:
            moy+=vectorizer_text.wv[word]
        X_pretext.append(moy/len(tweet))

    X_clean = []
    for ligne in X_pretext:
        X_clean.append(list(ligne)) 
    columns = [str(i) for i in range(len(X_clean[0]))]
    df_clean = pd.DataFrame(columns=columns, index=X.index)
    for i, j in enumerate(X.index):
        df_clean.loc[j] = X_clean[i]
    X = pd.concat([X, df_clean], axis = 1)
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

def preprocess_hashtags(X, train = True, vectorizer_hashtags = None):
    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags_count = hashtags.apply(lambda x : len(x))
    X['hashtag_count'] = hashtags_count
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

def add_NER(X):
    NER = spacy.load('fr_core_news_sm')
    X_ner = X['text'].apply(lambda text : text.split(' ')).apply(lambda lists : lists.apply(lambda word: str(word.label_))).apply(lambda x : ' '.join(x))
    vectorizer_text = TfidfVectorizer(max_features=150, stop_words=stopwords.words('french'))
    X_ner = pd.DataFrame(vectorizer_text.fit_transform(X_ner).toarray(), index = X.index)
    return X



def add_variables(X, train, vectorizer_text = None, vectorizer_hashtag = None):
    X["len_text"] = X["text"].apply(lambda x: len(x))

    X, vectorizer_text = preprocess_text(X, train, vectorizer_text)
    X = preprocess_time(X)
    X, vectorizer_hashtag = preprocess_hashtags(X, train, vectorizer_hashtag)
    X = preprocess_urls(X)
    X = add_sentiments(X)
    return X, vectorizer_text, vectorizer_hashtag

def select_columns(X):
    X = X.drop(['text','mentions','urls','hashtags', 'timestamp','TweetID'], axis = 1)
    return X


def preprocessing(X, train, vectorizer_text = None, vectorizer_hashtag = None):
    X, vectorizer_text, vectorizer_hashtag = add_variables(X, train, vectorizer_text, vectorizer_hashtag)
    X = select_columns(X)
    return X, vectorizer_text, vectorizer_hashtag

def load_train_data(test):
    # Load the training data
    train_data = pd.read_csv("train.csv")

    if test == True:
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_data['retweets_count'], test_size=0.3)
        X_test = X_test.drop(['retweets_count'], axis=1)
        X_train = X_train.drop(['retweets_count'], axis=1)
    else :
        y_train = train_data['retweets_count']
        X_train = train_data.drop(['retweets_count'], axis=1)
    
    # We preprocess the data
    X_train, vectorizer_text, vectorizer_hashtag = preprocessing(X_train, train = True)
    if test == True: 
        X_test, vectorizer_text, vectorizer_hashtag = preprocessing(
                                X_test, 
                                train = False, 
                                vectorizer_text = vectorizer_text,
                                vectorizer_hashtag = vectorizer_hashtag,
                                )
        return X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtag

    else: return X_train, y_train, vectorizer_text, vectorizer_hashtag

def load_validation_data(vectorizer_text, vectorizer_hashtag):
    eval_data = pd.read_csv("evaluation.csv")
    X_eval, vectorizer_text, vectorizer_hashtag = preprocessing(eval_data, 
                        train=False, 
                        vectorizer_text= vectorizer_text,
                        vectorizer_hashtag = vectorizer_hashtag,
                        )
    return X_eval

def main_preprocessing():
    X_train, y_train, X_test, y_test, vectorizer_text = load_train_data(test=True)
    X, y, vectorizer_text = load_train_data (test=False)
    X_val = load_validation_data(
        vectorizer_text=vectorizer_text,
        )
    save_data('preprocessing4', X, y, X_train, y_train, X_test, y_test, X_val)
    
main_preprocessing()
# %%


# %%
