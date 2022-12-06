import pandas as pd
import os
import numpy as np

def load_data(name_folder):
    X_train = pd.read_csv('data/'+ name_folder +'/X_train.csv', index_col=0)
    X_test = pd.read_csv('data/'+ name_folder +'/X_test.csv', index_col=0)
    y_train = pd.read_csv('data/'+ name_folder +'/y_train.csv', index_col=0)
    y_test = pd.read_csv('data/'+ name_folder +'/y_test.csv', index_col=0)
    X = pd.read_csv('data/'+ name_folder +'/X.csv', index_col=0)
    y = pd.read_csv('data/'+ name_folder +'/y.csv', index_col=0)
    X_val = pd.read_csv('data/'+ name_folder +'/X_val.csv', index_col=0)
    return X, y, X_train, y_train, X_test, y_test, X_val

def load_data_numpy(name_folder):
    X_train = np.load('data/'+ name_folder +'/X_train.npy')
    X_test = np.load('data/'+ name_folder +'/X_test.npy')
    y_train = np.load('data/'+ name_folder +'/y_train.npy')
    y_test = np.load('data/'+ name_folder +'/y_test.npy')
    X = np.load('data/'+ name_folder +'/X.npy')
    y = np.load('data/'+ name_folder +'/y.npy')
    X_val = np.load('data/'+ name_folder +'/X_val.npy')
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

def save_data_numpy(name_folder, X, y, X_train, y_train, X_test, y_test, X_val):
    os.makedirs('data/' + name_folder, exist_ok=True)
    np.save('data/' + name_folder + '/X_train', X_train)
    np.save('data/' + name_folder + '/X_test', X_test)
    np.save('data/' + name_folder + '/y_train', y_train.to_numpy())
    np.save('data/' + name_folder + '/y_test', y_test.to_numpy())
    np.save('data/' + name_folder + '/X', X)
    np.save('data/' + name_folder + '/y', y.to_numpy())
    np.save('data/' + name_folder + '/X_val', X_val)