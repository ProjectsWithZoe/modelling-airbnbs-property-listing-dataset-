# modelling-airbnbs-property-listing-dataset-

A set of images and tabular data was downloaded to be used for this project from Airbnb. 
Firstly, the images were resized such that the height of every image was the same as the height of the smallest image in that particular folder.
Images not in RGB format were also removed and the new resized images were saved into their own new folders in a directory labelled as 'processed images'
This function was then stored in a if __name__ == '__main__' block to only run if its the main file and not an imported file.

Next, the tabular data was analysed and cleaned by firstly removing the empty rows containing no ratings. Next, the description column was turned into a list from a previous list of strings using the literal_eval method imported from ast.
For some of the numerical columns containing NA values, they were automatically switched to a default value of 1.
The code for cleaning this tabular data was stored in a clean_tabular_data method that runs in an if __name__ == '__main__' block.
A load_airbnb data method was created using only the numerical tabular data where a particular column is used as a label and the corresponding features (the remaining tabular data excluding the label) are also returned. 
This returns a tuple containing the label and features once called.

A modelling.py file was then created to load a CSV file, clean the data, and then perform linear regression using scikit-learn's SGDRegressor.

We use the clean_tabular_data function to clean the data and load_airbnb function from the tabular_data module to load the data as a tuple, selecting only the columns with numerical data. This data is then stored as X and y variables corresponding to the features and labels and then converted to a Numpy array.
Next, the train_test_split() function imported from sklearn.model_selection was used to split the data into training, validation, and testing sets with a test size of 25% of the total data and a random state of 42 for reproducibility. The new training set and validation set were also split using train_test_split() with the same test size and random state.
A SGDRegressor() model was then created with a tolerance of 1e-3, L2 regularization, and a random state of 42. The model was fitted on the training data using the fit() method. Predictions were made on the test set using the predict() method, and the mean squared error between the true labels and predicted labels was calculated using mean_squared_error() function.

The custom_tune_regression_model_hyperparameters() function takes in several parameters including the class of the model to be tuned, training, validation, and test sets, and a dictionary of hyperparameters to be tuned. The function initializes several variables such as best_score, best_params, and validation_RMSE. It then uses itertools.product() to create all possible combinations of the hyperparameters to be tuned, and for each combination, the model is created and fitted on the training data using the hyperparameters. The score of the model on the validation set is calculated using model.score(), and the root mean squared error between the true labels and predicted labels on the validation set is calculated using mean_squared_error(). If the RMSE on the validation set is smaller than the previous smallest RMSE, the RMSE, score, and best parameters are updated. Finally, a dictionary of metrics including the best R2 score and RMSE is returned along with the best model and its best parameters.

The tune_regression_model_hyperparameters() function takes in the class of the model to be tuned and a dictionary of hyperparameters to be tuned.
The function initializes the model using the model class, and then performs a grid search using GridSearchCV() to find the best combination of hyperparameters based on the provided hyperparameter dictionary. The GridSearchCV() function automatically performs cross-validation and returns the best parameters found during the search. The best parameters found during the grid search are returned from the function.

The save_model() function takes in the trained model, metrics obtained during training, hyperparameters used during training, and a folder path where the model should be saved. The function first creates the directory specified in the folder parameter if it doesn't already exist. It then saves the trained model using joblib.dump(), and saves the hyperparameters and metrics in separate JSON files using the json.dump() function.

The evaluate_all_models function takes a list of regression model classes as input, tunes the hyperparameters of each model using tune_regression_model_hyperparameters function, fits the best parameters to the model, evaluates the model performance on the validation set, saves the model, hyperparameters, and metrics data in a new folder for each model. Finally, it returns the last evaluated model, its metrics data, the corresponding folder, and the tuned hyperparameters.

The find_best_model function searches through the saved models' directories to find the model with the lowest RMSE on the validation set. It loads the metrics data and hyperparameters from the saved files of each model and returns the model name, its metrics data, the corresponding hyperparameters, and the RMSE of the best-performing model. 
