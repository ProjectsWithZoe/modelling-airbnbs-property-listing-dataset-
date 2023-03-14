import tabular_data
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import json
from joblib import dump

file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)

df= tabular_data.clean_tabular_data(df=raw_df)
tup = tabular_data.load_airbnb(df,label='Price_Night')
#X = tup[1] #features
#y = tup[0] #labels
nums = ['float64', 'int64'] 
X_nums = tup[1].select_dtypes(include = nums)

def scale_data(features, labels):
    #converts features and labels to np array and applies scaling

    X = np.array(X_nums)
    y = np.array(tup[0])
    X = scale(X)

    return X,y
X,y = scale_data(X_nums, tup[0])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
model = SGDRegressor(tol=1e-3, penalty='l2', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
#print('Mean squared error:', mse)

#predictions for training and test sets
#y_train_pred = model.predict(X_train)
#y_test_pred = model.predict(X_test)

#print(y_train_pred, y_test_pred)

#rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
#rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

#r2_train = r2_score(y_train, y_train_pred)
#r2_test = r2_score(y_test, y_test_pred)

#print(model.get_params())

hyperparameters = {
    'penalty': ['l1', 'l2'],
    'fit_intercept': [True, False],
    'alpha': [0.001,0.1,1,10],
    'max_iter':[10,100,200,500,1000, 10000, 100000, 1000000]
    }


def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters):
    best_score = 0
    best_params = {}
    validation_RMSE = float('inf')

    for values in itertools.product(*hyperparameters.values()):
        params = dict(zip(hyperparameters.keys(), values))
        model = SGDRegressor()
        model.set_params(**params)

        model.fit(X_train, y_train)
        score = model.score(X_val,y_val)
        y_val_pred = model.predict(X_val)
        
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

        #print(rmse_val)
        if rmse_val < validation_RMSE:
            validation_RMSE = rmse_val
            best_score = score
            best_params = params
    metrics = {'R2': best_score , 'RMSE': validation_RMSE}

    print (model, best_params, metrics)
    return model, best_params, metrics

#custom_tune_regression_model_hyperparameters(model_class=SGDRegressor, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, hyperparameters=hyperparameters)
def tune_regression_model_hyperparameters(model_class, hyperparameters):
    model = SGDRegressor()
    grid_search = GridSearchCV(model, hyperparameters)
    grid_search.fit(X, y)
    params = model.get_params()

    print(grid_search.best_params_)
    print(grid_search.best_score_)

#tune_regression_model_hyperparameters(SGDRegressor, hyperparameters)

def save_model(model, metrics, folder='models/regression/linear_regression'):
    os.makedirs(folder, exist_ok=True)
    dump(model, os.path.join(folder, 'model.joblib'))

    with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)

    with open(os.path.join(folder, 'metrics.json'), 'w') as f:
        json.dump(metrics,f)