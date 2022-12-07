**Predicting the number of retweets a tweet made during the 2022 presidential election will have.**

1. **Code structure**

We started by trying to understand the data thanks to Exploratory Data Analysis which allowed us to have an idea of the influence of the different variables provided (and those we constructed) on the value of y.

To recover the data, we just have to launch the preprocessing.py file which creates a data folder where the modified features (X_train, X_test, X_val and X) will be saved. X corresponds to the modified features of train.csv, i.e. the union of X_train and X_test. The preprocessing calls functions of nlp.py which allow to treat the variable 'text' in an efficient way thanks to Natural Language Processing. In test_nlp.py we have tried different approaches to process the text of these tweets.

For the training we tried different models which are separated in different files: 
- model_rfr.py: there are two models namely custom_model which consists of both a RandomForestClassifier to predict whether a tweet will have retweets or not, and if a tweet is predicted to have retweets then an LGBMRegressor predicts the number of retweets the tweet will have. The RandomForestRegressor (function rfr) is the model used when we run the code as described in the second part.
- model_nn.py: a Neural Network with unsatisfactory results, it might have been interesting to investigate this further; 
- model_xgboost.py : an XGBRegressor whose parameters have been optimized thanks to GridSearchCV.
- train.py : in this file we try different types of regression models such as Support Vector Regression, KNeighorsRegressor, RandomForestRegressor.

2. **To get a prediction thanks to our code**

**Firts method**

- Just run the main.ipynb file that will create a file with the retweet prediction in pred/predictions.csv.

**Second method**

- Step 1: **Run preprocessing.py**. This will create files with the modified data.  You need to uncomment the last line.
- Step 2: **Run prediction.py**. This file calls the ... model and returns a .csv file predicting the number of retweets for the data in evaluation.csv.  You need to uncomment the last line.
- Step 3: **Get the predictions in pred/predictions.csv**
