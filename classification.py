from sklearn.preprocessing import LabelEncoder, scale
import tabular_data, modelling
import pandas as pd
import numpy as np
import tabular_data
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import itertools
import modelling

file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)
df = tabular_data.clean_tabular_data(raw_df)

labels, features = tabular_data.load_airbnb(df, 'Category')
num_features = features.select_dtypes(include = modelling.nums)
le = LabelEncoder()
le.fit(df['Category'])
df['Category_encoded'] = le.transform(df['Category'])

X = np.array(num_features)
y = np.array(df['Category_encoded'])
X= scale(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
dtr_model = DecisionTreeClassifier()
dtr_model.fit(X_train, y_train)
dtr_pred = dtr_model.predict(X_test)
train_pred = dtr_model.predict(X_train)
dtr_score = dtr_model.score(X_val, y_val)

'''Metrics'''
accuracy_test = accuracy_score(y_test, dtr_pred)
accuracy_train = accuracy_score(y_train, train_pred)

recall_test = recall_score(y_test, dtr_pred, average='macro')
recall_train = recall_score(y_train, train_pred, average = 'macro')

precision_test = precision_score(y_test, dtr_pred, average='macro')
precision_train = precision_score(y_train, train_pred, average='macro')

f1_test = f1_score(y_test, dtr_pred, average='macro')
f1_train = f1_score(y_train, train_pred, average = 'macro')

''' HYPERPARAMETERS'''
dtc_hyperparameters = {
    'criterion':['gini', 'entropy', 'log_loss'],
    'splitter':['best', 'random'],
    'max_features':['auto', 'sqrt', 'log2']
}
knn_hyperparameters = {
    'n_neighbors': [2,5,8,10,100],
    'weights':['uniform', 'distance', None],
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
}
rfc_hyperparameters = {
    'n_estimators': [1,10,30,50,70,100],
    'criterion': ['gini', 'entropy', 'log_loss'],
    #'max_depth':[None, 1,5,10,50,100],
    'max_features' :['sqrt', 'log2', None],
    #'max_samples': [0.01, 0.1, 1, 10, 100, None]
}

gbc_hyperparameters = {
    'loss':['log_loss'],
    'learning_rate' : [0.1, 0.5, 0.9],
    'criterion': ['friedman_mse', 'squared_error']
    }


def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters):
    best_params = {}
    validation_accuracy = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    for values in itertools.product(*hyperparameters.values()):
        params = dict(zip(hyperparameters.keys(), values))
        model = model_class()
        model.set_params(**params)

        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred, average = 'macro')
        precision = precision_score(y_val, y_val_pred, average = 'macro')
        f1_score = f1_score(y_val, y_val_pred, average = 'macro')

        if accuracy > validation_accuracy:
            validation_accuracy = accuracy
            best_params = params
            best_f1 = f1_score
            best_precision = precision
            best_recall = recall

        metrics = (f'Validation accuracy: {validation_accuracy}, Recall : {best_recall}, F1 score: {best_f1}, Precision: {best_precision}')
            
    return (model, best_params, metrics)
model_classes = [DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier]

def evaluate_all_models(model_classes, task_folder):
    for model_class in model_classes:
        model = model_class()
        folder = task_folder+model_class.__name__
        if model_class == DecisionTreeClassifier:
            hyperparameters = dtc_hyperparameters
        elif model_class == RandomForestClassifier:
            hyperparameters = rfc_hyperparameters
        elif model_class == KNeighborsClassifier:
            hyperparameters = knn_hyperparameters
        else:
            hyperparameters = gbc_hyperparameters
        tuned_params = modelling.tune_regression_model_hyperparameters(model_class, hyperparameters)

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
        modelling.save_model(model=model, metrics=metrics, hyperparameters=tuned_params, folder=folder)
    print (model,metrics,folder, tuned_params)
    return model,metrics,folder, tuned_params

main_folder = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/models/classification'
modelling.find_best_model(main_folder)
    