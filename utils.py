import pandas as pd
import os

def load_data(name_folder):
    X_train = pd.read_csv('data/'+ name_folder +'/X_train.csv', index_col=0)
    X_test = pd.read_csv('data/'+ name_folder +'/X_test.csv', index_col=0)
    y_train = pd.read_csv('data/'+ name_folder +'/y_train.csv', index_col=0)
    y_test = pd.read_csv('data/'+ name_folder +'/y_test.csv', index_col=0)
    X = pd.read_csv('data/'+ name_folder +'/X.csv', index_col=0)
    y = pd.read_csv('data/'+ name_folder +'/y.csv', index_col=0)
    X_val = pd.read_csv('data/'+ name_folder +'/X_val.csv', index_col=0)
    return X, y, X_train, y_train, X_test, y_test, X_val

def save_data(name_folder, X, y, X_train, y_train, X_test, y_test, X_val):
    os.makedirs('data/' + name_folder , exist_ok=True)  
    X_train.to_csv('data/' + name_folder +'/X_train.csv')
    X_test.to_csv('data/' + name_folder +'/X_test.csv')
    X_val.to_csv('data/' + name_folder +'/X_val.csv')
    X.to_csv('data/' + name_folder +'/X.csv')
    y_train.to_csv('data/' + name_folder +'/y_train.csv')
    y_test.to_csv('data/' + name_folder +'/y_test.csv')
    y.to_csv('data/' + name_folder +'/y.csv')