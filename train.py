# %%
import numpy as np
from sklearn.metrics import mean_absolute_error

# from preprocessing import load_train_data
# X_train, y_train, X_test, y_test, vectorizer_text, vectorizer_hashtags, std_clf = load_train_data(test=True)
# np.save('data/array/X_train', X_train)
# np.save('data/array/X_test', X_test)
# np.save('data/array/y_train', y_train.to_numpy())
# np.save('data/array/y_test', y_test.to_numpy())

X_train = np.load('data/array/X_train.npy')
X_test = np.load('data/array/X_test.npy')
y_train = np.load('data/array/y_train.npy')
y_test = np.load('data/array/y_test.npy')

# from model import train_custom_model
# reg = train_custom_model(X_train, y_train)
# y_pred = reg.predict(X_test)
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))

from model import train_nnrf, get_normal_counter
regr, reg = train_nnrf(X_train, y_train)
nn_y_test_predict = regr.predict(np.log(X_test + 1))
rf_test_predict = reg.predict(X_test)
y_pred = get_normal_counter(nn_y_test_predict + rf_test_predict, logarithm="e")
y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))

# from model import train_lrrf
# regr, reg = train_lrrf(X_train, y_train)
# lr_y_pred_predict = reg.predict(np.log(X_test + 1))
# rf_pred_predict = reg.predict(X_test)
# y_pred = get_normal_counter(lr_y_pred_predict + rf_pred_predict, logarithm="e")
# y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
# print("Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))

