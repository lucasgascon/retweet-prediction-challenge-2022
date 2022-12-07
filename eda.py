#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from verstack.stratified_continuous_split import scsplit # pip install verstack

seed = 12

# Load the training data
train_data = pd.read_csv("train.csv")
y = train_data['retweets_count']
X = train_data.drop(['retweets_count'], axis=1)
X.head()
#%%

print(X.dtypes)
#%%

for col in X.select_dtypes('int'):
    plt.figure()
    X_col_q = X[col][X[col] < np.quantile(X[col],0.9)]
    sns.distplot(X_col_q)

# %%

for col in X.select_dtypes('int'):
    plt.figure()
    plt.scatter(X[col],y)
    plt.title(X[col].name)
    plt.show()

#%%

verified = y[np.array(X['verified'] == 1)]
not_verified = y[np.array(X['verified'] == 0)]

# plt.figure()
# sns.distplot(verified[verified < np.quantile(verified, 0.9)], label='verified')
# sns.distplot(not_verified[not_verified < np.quantile(not_verified, 0.9)], label='not verified')
# plt.legend()

plt.figure()
sns.distplot(verified, label='verified')
sns.distplot(not_verified, label='not verified')
plt.xlim(0,2000)
plt.ylim(0,0.0015)
plt.legend()

print(y[verified].mean())
print(y[not_verified].mean())

#%% 

print(np.quantile(y[verified], [0.25, 0.5, 0.75]))
print(np.quantile(y[not_verified], [0.25, 0.5, 0.75]))
#%%


# sns.pairplot(X.select_dtypes('int'))

# %%

sns.clustermap(train_data.select_dtypes('int').corr())
# %%
