#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from verstack.stratified_continuous_split import scsplit # pip install verstack

seed = 12
#%%

# Load the training data
train_data = pd.read_csv("train.csv")

X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)
X_train.head()
# %%

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# X_train_new = SelectKBest(chi2, k=4).fit_transform(X_train.select_dtypes('int'), y_train)
# X_train_new['text'] = X_train['text']
# print(X_train_new.head())

len_urls = X_train['urls'].map(lambda x : len(x))

len_urls.sort_values(ascending = False)

# %%



# %%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# configure to select all features
fs = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)