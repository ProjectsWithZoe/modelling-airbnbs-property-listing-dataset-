from sklearn.preprocessing import LabelEncoder, scale
import tabular_data, modelling
import pandas as pd
import numpy as np
import tabular_data
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import json
from joblib import dump




file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)
df = tabular_data.clean_tabular_data(raw_df)

labels, features = tabular_data.load_airbnb(df, 'Category')
num_features = features.select_dtypes(include = modelling.nums)
#print(num_features)
#print(labels)
le = LabelEncoder()
le.fit(df['Category'])
df['Category_encoded'] = le.transform(df['Category'])
#print((df['Category_encoded']).unique())
#print(df['Category_encoded'].shape)
#print(features.shape)

X = np.array(num_features)
y = np.array(df['Category_encoded'])
X= scale(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
dtr_model = DecisionTreeClassifier()
dtr_model.fit(X_train, y_train)
dtr_predictions = dtr_model.predict(X_test)
score = dtr_model.score(X_val, y_val)

print(dtr_predictions)
print(score)
