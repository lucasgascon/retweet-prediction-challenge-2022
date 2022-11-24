# %%

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

# seed = 12

import torch
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
if(torch.backends.mps.is_available() & torch.backends.mps.is_built()): 
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('device : ', device)

#%%
from preprocessing import load_train_data
X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags = load_train_data(test=True)

#%%
# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
#reg = GradientBoostingRegressor()
reg = RandomForestRegressor()
#reg = LinearRegression()


# from model import train_nnrf
# regr = train_nnrf(X_train, y_train)
# y_pred = regr.predict(X_test)
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))

#%%
# # We fit our model using the training data
reg.fit(X_train, y_train)
# # And then we predict the values for our testing set
y_pred = reg.predict(X_test)
# # We want to make sure that all predictions are non-negative integers
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))





