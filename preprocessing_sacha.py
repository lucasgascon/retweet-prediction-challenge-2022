#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from verstack.stratified_continuous_split import scsplit # pip install verstack
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from sklearn.pipeline import make_pipeline
from nlp import preprocess_text, preprocess_text_2

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
    X['hour'] = X['timestamp'].apply(lambda timestamp : int(datetime.fromtimestamp(timestamp / 1000).strftime("%H")))
    X['day'] = X['timestamp'].apply(lambda timestamp : datetime.fromtimestamp(timestamp / 1000).strftime("%a"))
    day = {'Mon':1, 'Tue':1, 'Wed':1, 'Thu':1, 'Fri': 0, 'Sat':0, 'Sun':1}
    hours = {0:1, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1, 18:1, 19:1, 20:1, 21:1, 22:1, 23:1} 
    X['day'] = X['day'].apply(lambda x : day[x])
    X['hour'] = X['hour'].apply(lambda x : hours[x])
    return X


def hashtagvalue(x):
    if x<6:
        return 1
    return 0

def preprocess_hashtags(X, train = True, vectorizer_hashtags = None):
    hashtags = X['hashtags'].apply(lambda x : x[2:-2].split(','))
    hashtags.apply(lambda x : x.remove('') if '' in x else x)
    hashtags_count = hashtags.apply(lambda x : hashtagvalue(len(x)))
    X['hashtag_count'] = hashtags_count
    hashtags_text = hashtags.apply(lambda x : ' '.join(x))
    if train : 
        vectorizer_hashtags = TfidfVectorizer(max_features=50, stop_words=stopwords.words('french'))
        hashtags_text = pd.DataFrame(vectorizer_hashtags.fit_transform(hashtags_text).toarray(), index = hashtags.index)
    else : 
        hashtags_text = pd.DataFrame(vectorizer_hashtags.transform(hashtags_text).toarray(), index = hashtags.index)
    # X = pd.concat([X, hashtags_text], axis = 1)
    
    return X, vectorizer_hashtags

def urlcount(x):
    if x==0:
        return 0
    return 1

def preprocess_urls(X):
    urls = X['urls'].apply(lambda x : x[2:-2].split(','))
    urls.apply(lambda x : x.remove('') if '' in x else x)
    urls = urls.apply(lambda x : urlcount(len(x)))
    X['url_count'] = urls
    return X

def add_sentiments(X):
    sia = SentimentIntensityAnalyzer()
    X['polarity_scores_neg'] = X['text'].apply(lambda text : sia.polarity_scores(text)['neg'])
    X['polarity_scores_pos'] = X['text'].apply(lambda text : sia.polarity_scores(text)['pos'])
    X['polarity_scores_compound'] = X['text'].apply(lambda text : sia.polarity_scores(text)['compound'])
    return X

def len_text_value(x):
    if x<50:
        return 0.5
    if x<210:
        return 1
    return 0


def prepocess_range_datas(X):
    X['favorites_count']=X['favorites_count']-X['favorites_count']%10
    X['followers_count']=X['followers_count']-X['followers_count']%10
    X['statuses_count']=X['statuses_count']-X['statuses_count']%10
    X['friends_count']=X['friends_count']-X['friends_count']%10
    return X

def add_variables(X, train, vectorizer_text = None, vectorizer_hashtags = None):
    
    X, vectorizer_text = preprocess_text(X, train, vectorizer_text)
    # X, vectorizer_text = preprocess_text_2(X, train, vectorizer_text)

    X = preprocess_time(X)
    X, vectorizer_hashtags = preprocess_hashtags(X, train, vectorizer_hashtags)
    X = preprocess_urls(X)
    X = add_sentiments(X)
    X = prepocess_range_datas(X)
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
        
        # X_train, X_test, y_train, y_test = train_test_split(train_data, train_data['retweets_count'], test_size=0.3, random_state=42)
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
X, y, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data (test=False)
X_val = load_validation_data(
    vectorizer_text=vectorizer_text,
    vectorizer_hashtags=vectorizer_hashtags,
    std_clf = std_clf,
    )

os.makedirs('data2/csv2', exist_ok=True)  
X_train.to_csv('data2/csv2/X_train.csv')
X_test.to_csv('data2/csv2/X_test.csv')
X_val.to_csv('data2/csv2/X_val.csv')
X.to_csv('data2/csv2/X.csv')
y_train.to_csv('data2/csv2/y_train.csv')
y_test.to_csv('data2/csv2/y_test.csv')
y.to_csv('data2/csv2/y.csv')
