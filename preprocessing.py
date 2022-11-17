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

from nltk.corpus import stopwords


seed = 12
#%%

def preprocessing(X, train, vectorizer = None):
    # We set up an Tfidf Vectorizer that will use the top 100 tokens from the tweets. We also remove stopwords.
    # To do that we have to fit our training dataset and then transform both the training and testing dataset. 
    

    X_only_int = X.select_dtypes('int')

    if train == True:
        vectorizer = TfidfVectorizer(max_features=100, stop_words=stopwords.words('french'))
        X_text = pd.DataFrame(vectorizer.fit_transform(X['text']).toarray(), index = X.index)
    else : X_text = pd.DataFrame(vectorizer.transform(X['text']).toarray(), index = X.index)

    X_new = pd.concat([X_only_int, X_text], axis = 1)

    if train == True:
        return X_new, vectorizer
    else: return X_new