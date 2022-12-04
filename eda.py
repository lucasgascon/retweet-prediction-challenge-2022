#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from verstack.stratified_continuous_split import scsplit # pip install verstack

seed = 12
#%%
from preprocessing import load_train_data, load_validation_data
# Load the training data
# train_data = pd.read_csv("train.csv")
# y = train_data['retweets_count']
# X = train_data.drop(['retweets_count'], axis=1)
# evaluation_data = pd.read_csv("evaluation.csv")
# X_train, y_train, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data(test=False)
# X_val = load_validation_data(vectorizer_text, vectorizer_hashtags, std_clf)

X_train, y_train, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data(test = False)

# sns.clustermap(X.select_dtypes('int').corr())
# %%

df = X_train.copy()
df['y'] = y_train
df.head()

for i in range(99):
    df = df.drop([i], axis =1)
# %%

#%%

for col in df.columns:
    print(col)
    plt.scatter(X_train[col], y_train)
    plt.show()


# %%
