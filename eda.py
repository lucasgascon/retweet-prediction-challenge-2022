#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

seed = 12
#%%

# dir = 'array'
# X = np.load('data/' + dir + '/X.npy')
# y = np.load('data/' + dir + '/y.npy')

X = pd.read_csv('data/csv/X.csv')
X.head()
#%%
y = pd.read_csv('data/csv/y.csv',index_col=0)

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
