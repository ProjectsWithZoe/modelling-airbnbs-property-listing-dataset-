import tabular_data
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt
#import seaborn as sns
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
'''PARAMETERS'''
sgd_hyperparameters = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'fit_intercept': [True, False],
    'alpha': [0.001,0.1,1,10,100],
    'max_iter':[10,100,200,500,1000, 10000, 100000, 1000000]
    }

dtr_parameters = {
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter':['best', 'random'],
    'min_samples_leaf': [2, 1]            
}

gbr_parameters = {
    'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
    'learning_rate': [0.1,1,10],
    'n_estimators':[0.1,1,10,100,500],
    'subsample':[0.0,0.2,0.4,0.6,0.8,1.0],
    'criterion':['friedman_mse', 'squared_error'],
    'max_features':['auto', 'sqrt', 'log2']
}
rfr_parameters = {
    'n_estimators':[1,10,100, 1000],
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
}

'''MODEL CLASSES'''
regressor_model_classes = [DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor]

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters):
    best_score = 0
    best_params = {}
    validation_RMSE = float('inf')
    #hyperparameters = sgd_hyperparameters

    for values in itertools.product(*hyperparameters.values()):
        params = dict(zip(hyperparameters.keys(), values))
        model = model_class()
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
'''Get best params with GridSearchCV'''
def tune_regression_model_hyperparameters(model_class, hyperparameters):
    model = model_class()
    grid_search = GridSearchCV(model, hyperparameters)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    return best_params

def save_model(model, metrics, hyperparameters, folder):
    os.makedirs(folder, exist_ok=True)
    dump(model, os.path.join(folder, 'model.joblib'))
    
    with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)

    with open(os.path.join(folder, 'metrics.json'), 'w') as f:
        json.dump(metrics,f)


def evaluate_all_models(model_classes, task_folder):
    for model_class in model_classes:
        model = model_class()
        folder = task_folder+model_class.__name__
        if model_class == DecisionTreeRegressor:
            hyperparameters = dtr_parameters
        elif model_class == RandomForestRegressor:
            hyperparameters = rfr_parameters
        else:
            hyperparameters = gbr_parameters
        #tune parameters using model class
        tuned_params = tune_regression_model_hyperparameters(model_class, hyperparameters)

        #set best parameters as parameters
        tuned_model = model.set_params(**tuned_params)

        #fit new parameters to model
        tuned_model.fit(X_train, y_train)

        #evaluate model performance
        y_val_pred = tuned_model.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        accuracy = accuracy_score(y_val, y_val_pred)
        score = tuned_model.score(X_val, y_val)
        metrics = {'RMSE': rmse_val, 'Accuracy': accuracy, 'Score': score }
        print (model,metrics,folder, tuned_params)
        save_model(model=model, metrics=metrics, hyperparameters=tuned_params, folder=folder)
    print (model,metrics,folder, tuned_params)
    return model,metrics,folder, tuned_params

def find_best_model(main_folder):
    lowest_rmse = 200
    lowest_rmse_dirs = set()
    #main_folder = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/models/regression'
    #def find_best_model():
    for dirpath, dirnames, filenames in os.walk(main_folder):
        model= os.listdir(main_folder)
        for filename in filenames:
            metrics_json = os.path.join(dirpath, filename)
            if metrics_json.endswith('metrics.json'):
                with open (metrics_json, 'r') as f:
                    mets = json.load(f)
                    if mets['RMSE'] < lowest_rmse:
                        lowest_rmse = mets['RMSE']
                        lowest_rmse_dir = {dirpath}
                        lowest_rmse_dirs.clear()
                        
                    elif mets['RMSE'] == lowest_rmse:
                        lowest_rmse_dirs.add(dirpath)
    #print(highest_value, highest_dir) 
    #print(type(main_folder))
    lowest_rmse_dir = ', '.join(lowest_rmse_dir)
    model = os.path.basename(lowest_rmse_dir)
    for items in os.listdir(lowest_rmse_dir):
        
        json_files = os.path.join(lowest_rmse_dir, items)
        if json_files.endswith('metrics.json'):
            with open(json_files, 'r') as f:
                metrics_data = json.load(f)
        if json_files.endswith('hyperparameters.json'):
            with open(json_files, 'r') as g:
                hyperparameters_data = json.load(g)
    print (f'Model: {model}, Lowest RMSE: {lowest_rmse}, Metrics Data: {metrics_data}, Best Parameters: {hyperparameters_data}')
    return model, lowest_rmse, metrics_data, hyperparameters_data

if __name__ == "__main__":
    task_folder = 'models/regression/'
    evaluate_all_models(model_classes=regressor_model_classes, task_folder=task_folder )
    main_folder = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/models/regression'
    find_best_model(main_folder)
            
