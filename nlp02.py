#%%
# Import des bibliothèques
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

#%%
dir = "csv"
X_train_temp = pd.read_csv('data/' + dir + '/X_train.csv',index_col = None)
X_test_temp = pd.read_csv('data/' + dir + '/X_test.csv',index_col = None)
y_train = pd.read_csv('data/' + dir + '/y_train.csv', index_col=0)
y_test = pd.read_csv('data/' + dir + '/y_test.csv', index_col=0)



#%%
X_train = pd.DataFrame()
X_test = pd.DataFrame()

for i in range(100):
    X_train[str(i)]=X_train_temp[str(i)]
    X_test[str(i)]=X_test_temp[str(i)]
#%%
  
X_train_temp  
#%%    
vocab_size = 100
hidden_size = 100
embedding_size = 1

# Définition de la structure du RNN
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size))
model.add(LSTM(units=hidden_size))
model.add(Dense(units=vocab_size, activation='softmax'))

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Entraînement du modèle sur les données de texte
model.fit(X_train, y_train, epochs=10)
#%%
# Utilisation du modèle entraîné pour prédire des séquences de caractères
predictions = model.predict(X_test)