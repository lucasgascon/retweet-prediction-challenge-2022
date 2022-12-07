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
train_data = pd.read_csv("data/csv/X_train.csv")
#y = pd.read_csv("data/csv/y_train.csv")

# y = train_data['retweets_count']
# X = train_data.drop(['retweets_count'], axis=1)

df = train_data.copy()
#df['y'] = y
df.head()
#%%

sns.clustermap(df.select_dtypes('int').corr())

#%%

variable_to_study = ['retweets_count', 'favorites_count', 'followers_count', 'friends_count', 'verified' ]

var_mean = []
var_std = []
var_min = []
var_max = []

for variable in variable_to_study:
    var_mean.append(df[variable].mean())
    var_std.append(df[variable].std())
    var_min.append(df[variable].min())
    var_max.append(df[variable].max())

print(var_mean)
print(var_std)
print(var_min)
print(var_max)
#%%

def visualize_repartition(data, n_min, n_max, k=False, b=30):
    vec = train_data[data]
    sns.displot(vec[vec < n_max][n_min < vec], kde=k, bins=b)


#%%
visualize_repartition('month',-1,13)    
visualize_repartition('day',0,8)
visualize_repartition('hour',-1,25)

#%%
visualize_repartition('followers_count',1,5000)    

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
    
plt.bar(range(24),mean_RT_hour)
plt.ylabel('Average number of retweet')

plt.show()
# %%
mean_RT_days = []
for i in range(1, 8):
    nbr_RT_days = df['y'][df['day']==i].mean()
    mean_RT_days.append(nbr_RT_days)
    print(f'nombre moyen de RT le {i}e jour de la semaine : {nbr_RT_days}')
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']   
plt.bar(days,mean_RT_days)
plt.ylabel('Average number of retweet')
plt.show()


# %%
mean_RT_month = []
for i in range(1, 13):
    nbr_RT_month = df['y'][df['month']==i].mean()
    mean_RT_month.append(nbr_RT_month)
    print(f'nombre moyen de RT le {i}e jour de la semaine : {nbr_RT_month}')

monthes = ["Jan",'Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']
plt.bar(monthes,mean_RT_month)
plt.ylabel('Average number of retweet')

plt.show()
# %%
for i in range(len(df. axes[1])):
    print(df.iloc[:,i].name)
# %%
plt.scatter(df['favorites_count'], df['y'])
plt.xlabel('Favorite count')
plt.ylabel('Number of retweet')
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
plt.xlabel('Followers count')
plt.ylabel('Average number of retweet')
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['favorites_count']>10*i) & (df['followers_count']<=10*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
    print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.xlabel('Favorite count')
plt.ylabel('Average number of retweet')
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['statuses_count']>10*i) & (df['followers_count']<=10*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.xlabel('Statuses count')
plt.ylabel('Average number of retweet')
plt.show()
# %%
mean_RT_fc = []
for i in range(100):
    nbr_RT_fc = df['y'][(df['friends_count']>10*i) & (df['followers_count']<=10*(1+i))].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
    
plt.scatter(range(100),mean_RT_fc)
plt.xlabel('Friends count')
plt.ylabel('Average number of retweet')
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
   
plt.bar(range(16),mean_RT_fc)
plt.xlabel('Number of hashtag')
plt.ylabel('Average number of retweet')
plt.show()
# %%
mean_RT_fc = []
for i in range(16):
    nbr_RT_fc = df['y'][df['url_count']==i].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
   
plt.bar(range(16),mean_RT_fc)
plt.xlabel('Number of url')
plt.ylabel('Average number of retweet')
plt.show()
# %%
print(df['y'][df['url_count']==2].count())


# %%
df['url_count'][df['url_count']>1]=1

mean_RT_fc = []
for i in range(16):
    nbr_RT_fc = df['y'][df['url_count']==i].mean()
    mean_RT_fc.append(nbr_RT_fc)
   # print(f'nombre moyen de RT pour la tranche {i} : {nbr_RT_fc}')
   
plt.bar(range(16),mean_RT_fc)
plt.xlabel('Presence of an url')
plt.ylabel('Average number of retweet')
plt.show()

#Jen suis la
# %%

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
