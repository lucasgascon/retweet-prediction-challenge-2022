#%%

import pandas as pd
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from verstack.stratified_continuous_split import scsplit

train_data = pd.read_csv("train.csv")
X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)
text = X_train['text']

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.train(text, total_examples=247778, epochs=10)
#%%

import nltk
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

french_stopwords = nltk.corpus.stopwords.words('french')
lemmatizer = FrenchLefffLemmatizer()

def French_Preprocess_listofSentence(listofSentence):
    preprocess_list = []
    for sentence in listofSentence :
        sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])

        sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())

        tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)

        words_w_stopwords = [i for i in tokenize_sentence if i not in french_stopwords]

        words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)

        sentence_clean = ' '.join(w for w in words_lemmatize)

        preprocess_list.append(sentence_clean)

    return preprocess_list

french_text = text

french_preprocess_list = French_Preprocess_listofSentence(french_text)

#%%
print(text[349808])

print(french_preprocess_list[1])

#%%

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
vectorizer_text = TfidfVectorizer(max_features=120, stop_words=stopwords.words('french'))





# %%

def tagged(text_l):
    for i, list_of_words in enumerate(text_l):
            yield TaggedDocument(list_of_words, [i])

text_list = X['text'].apply(lambda x : x.split(' '))
data_for_train = list(tagged(text_list))
vectorizer_text = Doc2Vec(vector_size = 100, min_count = 3, alpha = 0.025, min_alpha = 0.025)
vectorizer_text.build_vocab(data_for_train)
vectorizer_text.train(data_for_train, total_examples = vectorizer_text.corpus_count, epochs = 1)
text = X['text'].apply(lambda x : x.split(' '))
X_pretext = text.apply(lambda x : vectorizer_text.infer_vector(x))

#%%


#%%

X_text = pd.DataFrame(X_pretext, index = X.index)


X_text

#%%

def preprocess_text(X, train = True, vectorizer_text = None):
    if train == True:
        text_list = X['text'].apply(lambda x : x.split(' '))
        data_for_train = list(tagged(text_list))
        vectorizer_text = Doc2Vec(vector_size = 100, min_count = 3, alpha = 0.025, min_alpha = 0.025)
        vectorizer_text.build_vocab(data_for_train)
        vectorizer_text.train(data_for_train, total_examples = vectorizer_text.corpus_count, epochs = 1)
        text = X['text'].apply(lambda x : x.split(' '))
        X_pretext = text.apply(lambda x : vectorizer_text.infer_vector(x).toarray())
        X_text = pd.DataFrame(X_pretext, index = X.index)

        #vectorizer_text = TfidfVectorizer(max_features=120, stop_words=stopwords.words('french'))
        #X_text = pd.DataFrame(vectorizer_text.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer_text.transform(X['text']).toarray(), index = X.index)
    X = pd.concat([X, X_text], axis = 1)
    return X, vectorizer_text


X_pretext, v = preprocess_text(X,train=True)
#%%
X_pretext.head()
