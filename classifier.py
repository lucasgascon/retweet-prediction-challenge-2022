#%%

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import preprocessing as pr
import test_nlp

#%%

X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags, std_clf = pr.load_train_data(test=True, preprocess_text = test_nlp.preprocess_text)
model_pipe = [LogisticRegression(solver='liblinear'), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), GaussianNB()]

#%%
y_train_bis = []
y_test_bis = []
for y in y_train:
    if y == 0:
        y_train_bis.append(0)
    else:
        y_train_bis.append(1)
for y in y_test:
    if y == 0:
        y_test_bis.append(0)
    else:
        y_test_bis.append(1)

y_train = y_train_bis
y_test = y_test_bis


#%%
model_list = ['logistic regr', 'svm', 'knn', 'decision tree', 'random forest', 'naive bayes']
acc_list = []
auc_list = []
cm_list = []

for model in model_pipe:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_list.append(metrics.accuracy_score(y_test, y_pred))
    fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
    auc_list.append(round(metrics.auc(fpr, tpr), 2))
    cm_list.append(confusion_matrix(y_test, y_pred))

#%%
fig = plt.figure(figsize = (18, 10))
for i in range(len(cm_list)):
    cm = cm_list[i]
    model = model_list[i]
    sub = fig.add_subplot(2, 3, i+1).set_title(model)
    cm_plot = sns.heatmap(cm, annot = True, cmap='Blues_r')
    cm_plot.set_xlabel('predicted value')
    cm_plot.set_ylabel('actual value')

#%%
results = pd.DataFrame({'Model':model_list, 'accuracy':acc_list, 'AUC':auc_list})
results