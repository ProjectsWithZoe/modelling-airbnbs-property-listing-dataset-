import tabular_data
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)

df= tabular_data.clean_tabular_data(df=raw_df)
tup = tabular_data.load_airbnb(df,label='Price_Night')
X = tup[1] #features
y = tup[0] #labels
nums = ['float64', 'int64'] 
X = X.select_dtypes(include = nums)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse);
#plot=sns.scatterplot(x=X, y=y)
#print(plot)
