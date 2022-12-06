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

X = pd.read_csv('data/csv/X.csv', index_col = None)
y = pd.read_csv('data/csv/y.csv', index_col=0)

df = X.copy()
df['y'] = y
df = df.drop(['Unnamed: 0'], axis =1)
df.head()
#%%


for i in range(99):
    df = df.drop([str(i)], axis =1)

df.head()

sns.clustermap(df.select_dtypes('int').corr())

for col in df.columns:
    if col != 'y':
        print(col)
        plt.scatter(df[col], df['y'])
        plt.show()

# %%
nb_verified = sum(df['verified'])
nb_rt_rev = sum(df['y'][df['verified']==1])

nb_not_verified = sum(df['verified']==0)
nb_rt_not_rev = sum(df['y'])-nb_rt_rev

print(f'nombre moyen de RT si non verifié : {nb_rt_not_rev/nb_not_verified}')
print(f'nombre moyen de RT si verifié : {nb_rt_rev/nb_verified}')
# %%
mean_RT_hour = []
for i in range(24):
    nbr_RT_h = df['y'][df['hour']==i].mean()
    mean_RT_hour.append(nbr_RT_h)
    print(f'nombre moyen de RT a {i}h : {nbr_RT_h}')
    
plt.scatter(range(24),mean_RT_hour)
plt.show()
# %%
mean_RT_days = []
for i in range(1, 8):
    nbr_RT_days = df['y'][df['day']==i].mean()
    mean_RT_days.append(nbr_RT_days)
    print(f'nombre moyen de RT le {i}e jour de la semaine : {nbr_RT_days}')
    
plt.scatter(range(1, 8),mean_RT_days)
plt.show()


# %%
mean_RT_month = []
for i in range(1, 13):
    nbr_RT_month = df['y'][df['month']==i].mean()
    mean_RT_month.append(nbr_RT_month)
    print(f'nombre moyen de RT le {i}e jour de la semaine : {nbr_RT_month}')
    
plt.scatter(range(1, 13),mean_RT_month)
plt.show()
# %%
for i in range(len(df. axes[1])):
    print(df.iloc[:,i].name)
# %%
plt.scatter(df['favorites_count'], df['y'])
plt.show()
# %%
plt.scatter(df['followers_count'], df['y'])
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['followers_count']>100*i) & (df['followers_count']<=100*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
    print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['favorites_count']>10*i) & (df['followers_count']<=10*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
    print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['statuses_count']>10*i) & (df['followers_count']<=10*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['friends_count']>10*i) & (df['followers_count']<=10*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.show()
# %%
df['favorites_count_range']=df['favorites_count']-df['favorites_count']%10
# %%
sns.clustermap(df.select_dtypes('int').corr())

# %%
df['followers_count_range']=df['followers_count']-df['followers_count']%50
# %%
plt.scatter(df['hashtag_count'], df['y'])
plt.show()
# %%
mean_RT_fc = []
for i in range(16):
    nbr_RT_fc = df['y'][df['hashtag_count']==i].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
   
plt.scatter(range(16),mean_RT_fc)
plt.show()
# %%
mean_RT_fc = []
for i in range(16):
    nbr_RT_fc = df['y'][df['url_count']==i].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
   
plt.scatter(range(16),mean_RT_fc)
plt.show()
# %%
print(df['y'][df['url_count']==2].count())
# %%
plt.scatter(df['len_text'], df['y'])
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['len_text']>10*i) & (df['len_text']<=10*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.show()
# %%
plt.scatter(df['mention_count'], df['y'])
plt.show()
# %%
mean_RT_fc = []
for i in range(16):
    nbr_RT_fc = df['y'][df['mention_count']==i].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
   
plt.scatter(range(16),mean_RT_fc)
plt.show()
# %%
