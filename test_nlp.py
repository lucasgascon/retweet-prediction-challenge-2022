


#%%
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

train_data = pd.read_csv("train.csv")
y = train_data['retweets_count']
train_data.drop(columns='retweets_count')
X_train, X_test, y_train, y_test = train_test_split(train_data, y, random_state=42, test_size=0.3)


print()

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

def preprocess_text_3(X, train = True, vectorizer_text = None):

    if train == True:
        modelname = 'flaubert/flaubert_base_cased' 

        # Load pretrained model and tokenizer
        flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
        flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
        # do_lowercase=False if using cased models, True if using uncased ones
        text = X_train['text']
        print(text)
        text_vect = []
        i = 0
        for sent in text:
            text_vect.append(flaubert_vect(sent, flaubert_tokenizer, flaubert))
            print(i)
            i+=1
        #X_pretext = text.apply(lambda x : flaubert_vect(x, flaubert_tokenizer, flaubert))
        X_pretext = text_vect
        print(X_pretext)
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
    print(X)
    return X, vectorizer_text

def preprocess_text_4(X, train = True, vectorizer_text = None):

    if train == True:
        nlp_spacy = sp.load("fr_core_news_sm")
        text = X_train['text']
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
        flatten_df = flatten(pd.DataFrame(X_pretext), 'text')
        vectorize_array = np.reshape(np.array(flatten_df.values), (-1, 100))
        X_text = pd.DataFrame(vectorize_array, index = X.index)
 
    X = pd.concat([X, X_text], axis = 1)
    print(X)
    return X, vectorizer_text
        



preprocess_text_4(X_train)
# %%


# %%
