#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

seed = 12
#%%
from utils import load_data, load_data_numpy

X, y, X_train, y_train, X_test, y_test, X_val = load_data('preprocessing')
X.head()

#%%


df = X.copy()
df['y'] = y
df.head()
#%%

df = df.drop(['Unnamed: 0'], axis =1)
for i in range(100):
    df = df.drop([str(i)], axis =1)

df.head()

sns.clustermap(df.select_dtypes('int').corr())

for col in df.columns:
    if col != 'y':
        print(col)
        plt.scatter(df[col], df['y'])
        plt.show()

# %%

df.columns
# %%
