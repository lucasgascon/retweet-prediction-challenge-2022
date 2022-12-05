


#%%
from argparse import _MutuallyExclusiveGroup
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy_sentence_bert
from transformers import FlaubertModel, FlaubertTokenizer
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import spacy as sp
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

train_data = pd.read_csv("train.csv")
y = train_data['retweets_count']
train_data.drop(columns='retweets_count')
X_train, X_test, y_train, y_test = train_test_split(train_data, y, random_state=42, test_size=0.3)
# X_train = X_train[:100][:]
#print(X_train)
def flatten(df, col):
    col_flat = pd.DataFrame([[i, x] for i, y in df[col].apply(list).iteritems() for x in y], columns=['I', col])
    col_flat = col_flat.set_index('I')
    df = df.drop(col, 1)
    df = df.merge(col_flat, left_index=True, right_index=True)

    return df


def flaubert_vect(sentence, flaubert_tokenizer, flaubert):
  
    token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])

    last_layer = flaubert(token_ids)[0]
    cls_embedding = last_layer[:, 0, :]
    return cls_embedding.detach().numpy()[0]


def spacy_vect(sentence, nlp_spacy):
    
    doc = nlp_spacy(sentence)
    
    return doc.vector


def sbert_vect(sentence, model):
    model.encode('')



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
        vectorizer_text = TfidfVectorizer(max_features=120, stop_words=stopwords.words('french'))
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
        # flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
        # vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
        # X_text = pd.DataFrame(vectorize_array, index = X.index)
    else : 
        text = X['text'].apply(lambda x : x.split(' '))
        X_pretext = text.apply(lambda x : vectorizer_text.infer_vector(x))
    #     flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
    #     vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
    #     X_text = pd.DataFrame(vectorize_array, index = X.index)
    # X = pd.concat([X, X_text], axis = 1)

    return X_pretext, vectorizer_text


def preprocess_text_3(X, train = True, vectorizer_text = None):

    if train == True:
        modelname = 'flaubert/flaubert_base_cased' 

        # Load pretrained model and tokenizer
        flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
        flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
        # do_lowercase=False if using cased models, True if using uncased ones
        text = X['text']
        #print(text)
        text_vect = []
        i = 0
        for sent in text:
            text_vect.append(flaubert_vect(sent, flaubert_tokenizer, flaubert))
            #print(i)
            #i+=1
        #X_pretext = text.apply(lambda x : flaubert_vect(x, flaubert_tokenizer, flaubert))
        X_pretext = text_vect
        # print(X_pretext)
        # flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
        # vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
        # X_text = pd.DataFrame(vectorize_array, index = X.index)
    else : 
        text = X['text']
        #print(text)
        text_vect = []
        i = 0
        for sent in text:
            text_vect.append(flaubert_vect(sent, flaubert_tokenizer, flaubert))
            #print(i)
    #     flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
    #     vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
    #     X_text = pd.DataFrame(vectorize_array, index = X.index)

    # X = pd.concat([X, X_text], axis = 1)
    # print(X)
    return X_pretext, vectorizer_text

def preprocess_text_4(X, train = True, vectorizer_text = None, modele = 'base'):

    
    if modele == 'base':
        nlp_spacy = sp.load("fr_core_news_sm")
    else:
        nlp_spacy = spacy_sentence_bert.load_model('stsb_roberta_large')
    text = X['text']
    print(text)
    text_vect = []
    i = 0
    for sent in text:
        text_vect.append(spacy_vect(sent, nlp_spacy))
        print(i)
        i+=1
    #X_pretext = text.apply(lambda x : flaubert_vect(x, flaubert_tokenizer, flaubert))
    X_pretext = text_vect
    print(X_pretext)
    #     flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
    #     vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
    #     X_text = pd.DataFrame(vectorize_array, index = X.index)

    # X = pd.concat([X, X_text], axis = 1)
    # print(X)
    return X_pretext, vectorizer_text
        

def preprocess_text_5(X, train=True, vectorizer_text = None):
    
        
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text = X['text']
    print(text)
    text_vect = []
    i = 0
    for sent in text:
        text_vect.append(model.encode(sent))
        print(i)
        i+=1
    #X_pretext = text.apply(lambda x : flaubert_vect(x, flaubert_tokenizer, flaubert))
    X_pretext = text_vect
    print(X_pretext)
    #     flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
    #     vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
    #     X_text = pd.DataFrame(vectorize_array, index = X.index)
 
    # X = pd.concat([X, X_text], axis = 1)
    # print(X)
    return X_pretext, vectorizer_text
        

def preprocess_text_6(X, train = True, vectorizer_text = None):
    
    text_list = X['text'].apply(lambda x : x.split(' '))
    vectorizer_text = Word2Vec(vector_size=150, window=3, min_count=1, workers=-1)
    vectorizer_text.build_vocab(text_list)
    vectorizer_text.train(text_list, total_examples = vectorizer_text.corpus_count, epochs = 10)
    text = X['text'].apply(lambda x : x.split(' '))
    X_pretext = []
    for tweet in text:
        moy = np.array([0.0 for i in range(150)])
        for word in tweet:
            moy+=vectorizer_text.wv[word]
        X_pretext.append(moy/len(tweet))
    # flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
    # vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
    # X_text = pd.DataFrame(vectorize_array, index = X.index)

    # flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
    # vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
    # X_text = pd.DataFrame(vectorize_array, index = X.index)
    # X = pd.concat([X, X_text], axis = 1)
    # print(X_pretext[0])
    # print(X_pretext[0])
    # print(len(X_pretext))

    return X_pretext, vectorizer_text

def create_train_df(X, train = True, vectorizer_text = None, modele = 'base', function = preprocess_text_2):
    if function == preprocess_text:
        return function(X, train, vectorizer_text)

    X_pretext, vectorizer_text = function(X, train, vectorizer_text)
    X_clean = []
    for ligne in X_pretext:
        X_clean.append(list(ligne)) 
    columns = [str(i) for i in range(len(X_clean[0]))]
    df_clean = pd.DataFrame(columns=columns, index=X.index)
    for i, j in enumerate(X.index):
        df_clean.loc[j] = X_clean[i]
    
    X = pd.concat([X, df_clean], axis = 1)
    return X, vectorizer_text

create_train_df(X_train, function = preprocess_text_6)[0]
# %%

from gensim.test.utils import common_texts
from gensim.models import Word2Vec


text_list = X_train['text'].apply(lambda x : x.split(' '))
vectorizer_text = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
vectorizer_text.build_vocab(text_list)
vectorizer_text.train(text_list, total_examples = vectorizer_text.corpus_count, epochs = 1)
# %%
