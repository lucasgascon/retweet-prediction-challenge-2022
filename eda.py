#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

seed = 12
#%%
y_test = pd.read_csv('data7/csv/y_test.csv', index_col=0)
pred_rfr = np.load('pred/pred_rfr.npy')
pred_xgb = np.load('pred/pred_xgb.npy')

y_pred = (pred_rfr + pred_xgb)/2

print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))

# dir = 'array'
# X = np.load('data/' + dir + '/X.npy')
# y = np.load('data/' + dir + '/y.npy')

X = pd.read_csv('data6/csv/X.csv',index_col=0)
X.head()
#%%
y = pd.read_csv('data6/csv/y.csv',index_col=0)

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
