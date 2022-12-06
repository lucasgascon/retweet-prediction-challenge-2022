#%%

import pandas as pd
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from verstack.stratified_continuous_split import scsplit

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv("train.csv")
X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)
text = X_train['text']


def flatten(df, col):
    col_flat = pd.DataFrame([[i, x] for i, y in df[col].apply(list).iteritems() for x in y], columns=['I', col])
    col_flat = col_flat.set_index('I')
    df = df.drop(col, 1)
    df = df.merge(col_flat, left_index=True, right_index=True)

    return df


def tagged(text_l):
    for i, list_of_words in enumerate(text_l):
            yield TaggedDocument(list_of_words, [i])


def preprocess_text(X, train = True, vectorizer_text = None):
    if train == True:
        vectorizer_text = TfidfVectorizer(max_features=150, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer_text.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer_text.transform(X['text']).toarray(), index = X.index)
    X = pd.concat([X, X_text], axis = 1)
    return X, vectorizer_text


def preprocess_text_2(X, train = True, vectorizer_text = None):
    if train == True:
        text_list = X['text'].apply(lambda x : x.split(' '))
        data_for_train = list(tagged(text_list))
        vectorizer_text = Doc2Vec(vector_size = 100, min_count = 3, alpha = 0.025, min_alpha = 0.025)
        vectorizer_text.build_vocab(data_for_train)
        vectorizer_text.train(data_for_train, total_examples = vectorizer_text.corpus_count, epochs = 1)
        text = X['text'].apply(lambda x : x.split(' '))
        X_pretext = text.apply(lambda x : vectorizer_text.infer_vector(x))
        flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
        vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
        X_text = pd.DataFrame(vectorize_array, index = X.index)
    else : 
        text = X['text'].apply(lambda x : x.split(' '))
        X_pretext = text.apply(lambda x : vectorizer_text.infer_vector(x))
        flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
        vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
        X_text = pd.DataFrame(vectorize_array, index = X.index)
    X = pd.concat([X, X_text], axis = 1)
    return X, vectorizer_text


def create_vectorizer_text():
    train_data = pd.read_csv("train.csv")
    train_data = train_data.drop(columns='retweets_count')
    evaluation_data = pd.read_csv("evaluation.csv")
    X = pd.concat([train_data, evaluation_data], axis = 0)

    text_list = X['text'].apply(lambda x : x.split(' '))
    vectorizer_text1 = Word2Vec(vector_size=50, window=2, min_count=1, workers=-1)
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


vectorizer_text1 = create_vectorizer_text()
X = preprocess_text(X_train, vectorizer_text1)


# X_train_preprocess, vectorizer_text = preprocess_text_2(X_train,train=True)
# X_train_preprocess

# X_test_preprocess, vectorizer_text = preprocess_text_2(X_test, train=False, vectorizer_text= vectorizer_text)
# X_test_preprocess

# %%
X
# %%
