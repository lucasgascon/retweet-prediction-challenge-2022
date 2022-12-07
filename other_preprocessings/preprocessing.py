#%%

import pandas as pd
from verstack.stratified_continuous_split import scsplit # pip install verstack
from sklearn.preprocessing import StandardScaler
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import save_data

# vectorization of the text
def preprocess_text(X, train = True, vectorizer_text = None):
    if train == True:
        vectorizer_text = TfidfVectorizer(max_features=150, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer_text.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer_text.transform(X['text']).toarray(), index = X.index)
    X = pd.concat([X, X_text], axis = 1)
    return X, vectorizer_text

# process of the timestamp : erase the month and keep hour and day
def preprocess_time(X):
    #X['month'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%m")))
    X['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))
    X['day'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%a"))
    day = {'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri': 5, 'Sat':6, 'Sun':7}
    X['day'] = X['day'].apply(lambda x : day[x])
    X['am_pm'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%p"))
    am_pm = {'AM': 0, 'PM':1}
    X['am_pm'] = X['am_pm'].apply(lambda x : am_pm[x])
    return X

# keep the number of hashtag in the tweet
def preprocess_hashtags(X):
    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags_count = hashtags.apply(lambda x : len(x))
    X['hashtag_count'] = hashtags_count
    return X

# keep the number of URL in the tweet
def preprocess_urls(X):
    urls = X['urls'].apply(lambda x : x[2:-2].split(','))
    urls.apply(lambda x : x.remove('') if '' in x else x)
    urls = urls.apply(lambda x : len(x))
    X['url_count'] = urls
    return X

# add a pos, neg or compound sentiment to analyse the texte
def add_sentiments(X):
    sia = SentimentIntensityAnalyzer()
    X['polarity_scores_neg'] = X['text'].apply(lambda text : sia.polarity_scores(text)['neg'])
    X['polarity_scores_pos'] = X['text'].apply(lambda text : sia.polarity_scores(text)['pos'])
    X['polarity_scores_compound'] = X['text'].apply(lambda text : sia.polarity_scores(text)['compound'])
    return X

# preprocess all the data
def add_variables(X, train, vectorizer_text = None):
    X["len_text"] = X["text"].apply(lambda x: len(x))

    X, vectorizer_text = preprocess_text(X, train, vectorizer_text)
    X = preprocess_time(X)
    X = preprocess_hashtags(X)
    X = preprocess_urls(X)
    X = add_sentiments(X)
    return X, vectorizer_text

# drop some of the columns
def select_columns(X):
    X = X.drop(['text','mentions','urls','hashtags', 'timestamp','TweetID'], axis = 1)
    return X

# preprocess X
def preprocessing(X, train, vectorizer_text = None):
    X, vectorizer_text = add_variables(X, train, vectorizer_text)
    X = select_columns(X)
    return X, vectorizer_text


def load_train_data(test):
    # Load the training data
    train_data = pd.read_csv("train.csv")

    # If we are using the training data and that we want to test them we split the data in train and test to train the model
    # and to test it with the data it has never seen. 
    if test == True:
        X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)
        X_test = X_test.drop(['retweets_count'], axis=1)
        X_train = X_train.drop(['retweets_count'], axis=1)
    else :
        # If we just want to train the model, we train it on the all dataset to have more
        # data once we will test it on the evaluation dataset
        y_train = train_data['retweets_count']
        X_train = train_data.drop(['retweets_count'], axis=1)
    
    # We preprocess the data
    X_train, vectorizer_text= preprocessing(X_train, train = True)
    if test == True: 
        X_test, vectorizer_text = preprocessing(
                                X_test, 
                                train = False, 
                                vectorizer_text = vectorizer_text,
                                )
        return X_train, y_train, X_test, y_test, vectorizer_text

    else: return X_train, y_train, vectorizer_text

# loading for the evaluation data
def load_validation_data(vectorizer_text):
    eval_data = pd.read_csv("evaluation.csv")
    X_eval, vectorizer_text= preprocessing(eval_data, 
                        train=False, 
                        vectorizer_text= vectorizer_text,
                        )
    return X_eval

def main_preprocessing():
    X_train, y_train, X_test, y_test, vectorizer_text = load_train_data(test=True)
    X, y, vectorizer_text = load_train_data (test=False)
    X_val = load_validation_data(
        vectorizer_text=vectorizer_text,
        )
    save_data('preprocessing', X, y, X_train, y_train, X_test, y_test, X_val)
    
# main_preprocessing()
# %%