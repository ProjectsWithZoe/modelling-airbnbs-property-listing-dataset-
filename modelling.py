import tabular_data
import pandas as pd
import numpy as np

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)

df= tabular_data.clean_tabular_data(df=raw_df)
tup = tabular_data.load_airbnb(df,label='Price_Night')
X = tup[1] #features
y = tup[0] #labels
nums = ['float64', 'int64'] 
X = X.select_dtypes(include = nums)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
model = SGDRegressor(tol=1e-3, penalty='l2', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
#print('Mean squared error:', mse);
#plot=sns.scatterplot(x=X, y=y)
#print(plot)

#predictions for training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#print(y_train_pred, y_test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

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
    best_r2 = 0
    best_rmse = 0

    for values in itertools.product(*hyperparameters.values()):
        params = dict(zip(hyperparameters.keys(), values))
        model = SGDRegressor()
        model.set_params(**params)

        model.fit(X_train, y_train)
        score = model.score(X_val,y_val)
        y_val_pred = model.predict(X_val)

        #rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        #r2_train = r2_score(y_train_pred, y_train)
        r2_val = r2_score(y_val, y_val_pred)
        #print(rmse_train)
        #print(rmse_val)
        #print (rmse_val, r2_val) 
        if score > best_score:
            best_score = score
            best_params = params

    print(score, best_params)

        #if r2_val > best_r2:
            #best_r2 = r2_val
            #best_params = params

    #print(best_r2, best_params)
    #return best_score, best_params


custom_tune_regression_model_hyperparameters(model_class=SGDRegressor, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, hyperparameters=hyperparameters)
