<h1> Modelling using Airbnbs property listing dataset </h1>

A set of images and tabular data was downloaded to be used for this project from Airbnb.
Firstly, the images were resized such that the height of every image was the same as the height of the smallest image in that particular folder.
Images not in RGB format were also removed and the new resized images were saved into their own new folders in a directory labelled as 'processed images'
This function was then stored in a if **name** == '**main**' block to only run if its the main file and not an imported file.

Next, the tabular data was analysed and cleaned by firstly removing the empty rows containing no ratings. Next, the description column was turned into a list from a previous list of strings using the literal_eval method imported from ast.
For some of the numerical columns containing NA values, they were automatically switched to a default value of 1.
The code for cleaning this tabular data was stored in a clean_tabular_data method that runs in an if **name** == '**main**' block.
A load_airbnb data method was created using only the numerical tabular data where a particular column is used as a label and the corresponding features (the remaining tabular data excluding the label) are also returned.
This returns a tuple containing the label and features once called.

<h2> <b> Modelling </b> </h2>
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

<h2> <b> Classification </b> </h2>
The code defines a function named tune_classification_model_hyperparameters that takes in various inputs such as a classification model class, training, validation and testing data, and a dictionary of hyperparameters and their possible values. The function iterates through all possible combinations of hyperparameters, fits the model with each set of hyperparameters, and computes the validation accuracy, recall, precision, and F1-score for each.

The function returns the model with the best hyperparameters, along with the best hyperparameters themselves and the evaluation metrics. Finally, the function considers the validation accuracy to select the best hyperparameters.
Then, the code defines a function named evaluate_all_models that takes in a list of model classes and a task folder. The function iterates through each model class in the list, instantiates the model, and identifies the appropriate hyperparameters for the model class. It then tunes the hyperparameters using the tune_regression_model_hyperparameters function and finds the best hyperparameters for the model.

The function fits the model with the best hyperparameters and evaluates its performance on validation data by computing the root mean squared error, accuracy, and score. The function saves the best model, its hyperparameters, and performance metrics to a folder.

Finally, the function calls another function called find_best_model from the modelling module to find the best model from the saved models and its corresponding hyperparameters and performance metrics.

<h1> Pytorch.py </h1>
This code defines a PyTorch dataset for performing regression on Airbnb nightly rental prices.

The dataset is initialized with a pandas dataframe 'data', which contains the features and target variable for the regression problem.

The target variable, 'Price_Night', is extracted from the dataframe and stored in a PyTorch tensor.
The features are extracted from the dataframe by dropping the target variable column using the .drop() method, and are stored as a numpy array in the self.features attribute. The features are then standardized using the scale() function from the scikit-learn library.

The __getitem__() method returns a tuple containing the features and target variable for a given index.
The __len__() method returns the number of samples in the dataset.

This code then splits the original data into training, validation, and testing sets using the train_test_split() function. The train_test_split() function is called twice to obtain two sets of splits. First, the original data is split into a training set and a test set, where the test set comprises 20% of the original data. Then, the test set obtained from the previous step is further split into a validation set and a new test set. The validation set also comprises 50% of the test set obtained from the previous step. The random_state argument is set to 42 again for reproducibility.

Three AirbnbNightlyPriceRegressionDataset objects are then created from the three sets of data. Each dataset is created using the class defined earlier, which returns tuples of standardized features and target variables.
Finally, three DataLoader objects are created from the three datasets. The DataLoader objects are used to efficiently load the data in batches during model training and evaluation. Each DataLoader loads samples from its corresponding dataset and returns batches of 16 in this case with optional shuffling of the samples.

Get_nn_config() - This code defines a function get_nn_config() that reads a YAML file containing a dictionary of configuration parameters for a neural network model. The yaml.load() method is used to parse the YAML file into a Python dictionary, which is then returned by the function. The config_path variable contains the file path to the YAML file. The config_params dictionary defines two hyperparameters, hidden_size and learning_rate, and their corresponding search spaces. The hidden_size hyperparameter specifies the number of hidden units in the neural network, and the learning_rate hyperparameter specifies the learning rate used during optimization. The search spaces for both hyperparameters are specified as lists of possible values.

Generate_nn_configs() - This is a function that generates all possible combinations of configuration parameters, given a dictionary of parameter names and their possible values. It then creates a dictionary of configurations for each combination of parameters. The function first imports the product function from the itertools module. This function returns all possible combinations of the elements in each input iterable. The function then extracts the possible values of each parameter from the input dictionary of config_params and creates a list of all possible combinations of these values using the product function. Next, it creates a configuration dictionary for each combination of parameters. For each combination, the function loops through the keys in the input dictionary of config_params and assigns the corresponding value from the current combination of parameters to each key. The resulting dictionary of configuration is then appended to the configs list.
Finally, the function returns the list of configuration dictionaries.

Class LinearRegression(nn.Module) - This is a PyTorch nn.Module subclass defining a linear regression model. The class has two main methods: __init__() and forward().In the __init__() method, the model is defined by creating two linear layers (nn.Linear) and a ReLU activation function (nn.ReLU()). The input size of the first linear layer is input_size and the output size is specified by the hidden_size value from the input dictionary config. The second linear layer has an input size equal to the hidden_size and an output size of output_size. The dtype of the linear layers is set to torch.float64. The hidden_size and learning_rate values from the input dictionary config are also saved as attributes of the model for later use.
In the forward() method, the input tensor x is first cast to the same dtype as the weight tensor of the first linear layer using x.to(self.fc1.weight.dtype). Then, x is passed through the first linear layer (self.fc1), followed by the ReLU activation function (self.relu), and then through the second linear layer (self.fc2). The resulting tensor is returned as the output of the forward() method. 

train() - The train function takes a model, train_loader, val_loader, optimiser, and num_epochs as input. It trains the model for num_epochs epochs using the optimizer on the train_loader, computes the loss on the validation set, and saves the hyperparameters and metrics. The model is saved in the models/neural_networks/regression directory, named according to the current date and time. The function returns a dictionary of performance metrics.

find_best_nn - The find_best_nn function generates a set of configurations and then trains a linear regression model for each configuration using the train function. The hyperparameters and metrics for each trained model are saved in separate files. After training all the models, the function chooses the one with the lowest validation loss and saves its hyperparameters and metrics in a separate directory.

The best parameterized model had a hidden_size of 64 and learning rate of 0.001.
Its other metrics were :
  <ul> RMSE training loss: 4656.226372162173 </ul>
  <ul> RMSE validation loss: 1349.5539369233848 </ul>
  <ul> R2 score training: -0.9297545561113245 </ul>
  <ul> R2 score validation: 0.4920548079495457 </ul>
  <ul> training_duration: 0.05686497688293457</ul>
  <ul> inference_latency: 8.230209350585937e-07 </ul>
  
  This is a visualization of the best neural network.
![nn (1)](https://user-images.githubusercontent.com/118231395/234360572-3bb15cec-f846-4ea7-b450-00881db7fafd.svg)

<h1> Model testing </h1> 
Finally, the train and find_best_nn function were used to train a model where the label column was changed to 'beds' and the model was retrained and the metrics were returned. The best model metrics were :
{'RMSE_loss_train': 1.9740945318157754, 'RMSE_loss_val': 0.7474831117294297, 'R_squared_train': -1.1057008339368273, 'R_squared_val': -2.363674002782434, 'training_duration': 0.06733918190002441, 'inference_latency': 2.830028533935547e-07}, {'hidden_size': 32, 'learning_rate': 0.001})
The metrics with the other 16 parameters were :<img width="1146" src="https://user-images.githubusercontent.com/118231395/234649738-73327163-12a8-465d-9a81-0ca60e349a2c.png" align='left'>


