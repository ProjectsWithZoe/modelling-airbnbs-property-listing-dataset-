import torch
import numpy as np
import pandas as pd
import yaml
import tabular_data
import json, os
from joblib import dump
import time
from datetime import datetime
import json
import os
import time
import random

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)
data= tabular_data.clean_tabular_data(df=raw_df)
nums = ['float64', 'int64']
data = data.select_dtypes(include = nums)

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self,data):
        self.label = torch.tensor(data['Price_Night'].values).float().double()
        self.features = (data.drop('Price_Night', axis=1).values)
        self.features = scale(self.features)
        #self.features = self.features.values
    def __getitem__(self, index):
        return self.features[index], self.label[index].unsqueeze(0) 
    def __len__(self):
        return len(self.features)

#a = AirbnbNightlyPriceRegressionDataset(data)
#print(a.features.dtype)
#print(a.label.dtype)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_dataset = AirbnbNightlyPriceRegressionDataset(train_data)
val_dataset = AirbnbNightlyPriceRegressionDataset(val_data)
test_dataset = AirbnbNightlyPriceRegressionDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

def get_nn_config(file):
    with open (file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        #print(config)
        return config
config_path = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/nn_config.yaml'

config_params = {
    'hidden_size': [3, 10, 16, 32, 64],
    'learning_rate':[0.0001, 0.001]
}

def generate_nn_configs(config_params):
    # Generate all possible combinations of configuration parameters
    from itertools import product
    param_values = list(config_params.values())
    param_combinations = list(product(*param_values))
    
    # Create a config dictionary for each combination of parameters
    configs = []
    for params in param_combinations:
        config = {}
        for i, key in enumerate(config_params.keys()):
            config[key] = params[i]
        configs.append(config)
        
    return configs


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, config['hidden_size'], dtype=torch.float64)
        self.fc2 = nn.Linear(config['hidden_size'], output_size, dtype=torch.float64)
        self.relu = nn.ReLU()
        #self.input_size = 9
        #self.output_size = 1
        self.hidden_size = config['hidden_size']
        self.learning_rate = config['learning_rate']

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def save_model(model, hyperparameters, metrics, save_path):
    os.makedirs(save_path, exist_ok=True)
    if isinstance(model, nn.Module):
        
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
        with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparameters, f)
        with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)


def train(model, train_loader, val_loader,optimiser, num_epochs=10):
    #optimiser = torch.optim.SGD(model.parameters(), lr=model.learning_rate)
    writer = SummaryWriter()

    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
    save_path = os.path.join('models', 'neural_networks', 'regression', timestamp)
    #training
    for epoch in range(num_epochs):
        train_loss = 0.0
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            features, label = batch
            train_predictions = model(features)
            loss = F.mse_loss(train_predictions, label)
            train_loss +=loss.item()
            #print (train_loss)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            writer.add_scalar('loss', loss.item(), epoch*len(train_loader)+i)
        train_loss /= len(train_loader)
        
        writer.add_scalar('avg train loss', train_loss, epoch)
        training_duration = time.time() - start_time
        writer.add_scalar('training duration', training_duration, epoch)

        #validation
        avg_val_loss = 0.0
        with torch.no_grad():
            for j, val_batch in enumerate(val_loader):
                val_features, val_label = val_batch
                val_predictions = model(val_features)
                val_loss = F.mse_loss(val_predictions, val_label)
                avg_val_loss += val_loss.item()
                #print (val_loss)
                writer.add_scalar('val loss', val_loss.item(), epoch*len(val_loader)+i)
            avg_val_loss /=len(val_loader)
            writer.add_scalar('avg val loss', val_loss, epoch)

        # Calculate RMSE loss and R-squared score for training set
        train_rmse_loss = mean_squared_error(label.detach().numpy(), train_predictions.detach().numpy())
        train_r2_score = r2_score(label.detach().numpy(), train_predictions.detach().numpy())

        # Calculate RMSE loss and R-squared score for validation set
        val_rmse_loss = mean_squared_error(val_label.detach().numpy(), val_predictions.detach().numpy())
        val_r2_score = r2_score(val_label.detach().numpy(), val_predictions.detach().numpy())

        # Calculate inference latency
        num_samples = 1000
        model.input_size = 9
        #model.hidden_size = 3
        #model.output_size = 1
        features = torch.randn((num_samples, model.input_size))
        start_time = time.time()
        model(features)
        inference_latency = (time.time() - start_time) / num_samples
        writer.add_scalar('inference latency', inference_latency, epoch)

        # Save the model, hyperparameters, and performance metrics
        hyperparameters = {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            #'output_size': model.output_size,
            'learning_rate': model.learning_rate,
            #'hidden_layer_width': model.hidden_layer_width,
            #'model_depth': model.model_depth
        }
        metrics = {'RMSE_loss_train': train_rmse_loss, 'RMSE_loss_val': val_rmse_loss,
               'R_squared_train': train_r2_score, 'R_squared_val': val_r2_score,
               'training_duration': training_duration, 'inference_latency': inference_latency}
        
        return metrics
    
    # Create a new folder for the current date and time
    now = datetime.now()
    folder_name = 'models/neural_networks/regression/' + now.strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs(folder_name)

    #Save the model, hyperparameters, and metrics
    save_model(model,hyperparameters, metrics, save_path)
    #print(metrics)
    #print(f'Train_RMSE: {train_rmse_loss}, Val RMSE: {val_rmse_loss}')
    #print(f'Train_v2: {train_r2_score}, Val_v2: {val_r2_score}')

def find_best_nn(train_loader, val_loader, config_params):
    # Generate configurations
    configs = generate_nn_configs(config_params)
    
    # Train models with each configuration
    best_model = None
    best_metrics = None
    best_hyperparams = None
    best_val_loss = float('inf')
    
    for i in range(17):
        new_config = random.choice(configs)
        # Create model
        model = LinearRegression(input_size=9, output_size=1, config=new_config)
        
        # Create optimizer
        optimiser = torch.optim.SGD(model.parameters(), model.learning_rate)
        
        # Train model
        metrics = train(model, train_loader, val_loader, optimiser)

        #print(metrics)

        # Save hyperparameters and metrics
        save_dir = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/16nn/'
        save_model(model, new_config, metrics, f'{save_dir}model_{i}')
        best_save_dir = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/16nn/best_nn'
        #hyperparams_file = os.path.join(save_dir, f'hyperparameters_{i}.json')
        #with open(hyperparams_file, 'w') as f:
         #   json.dump(new_config, f)
            
        #metrics_file = os.path.join(save_dir, f'metrics_{i}.json')
        #with open(metrics_file, 'w') as f:
        #    json.dump(metrics, f)
            
        # Save model if it performs better on the validation set
        val_loss = metrics['RMSE_loss_val']
        if val_loss < best_val_loss:
            best_model = model
            best_metrics = metrics
            best_hyperparams = new_config
            best_val_loss = val_loss
            #torch.save(model.state_dict(), os.path.join(best_save_dir, 'best_model.pth'))
            save_model(best_model, best_hyperparams, best_metrics, best_save_dir)
            
    return best_model, best_metrics, best_hyperparams

find_best_nn(train_loader, val_loader, config_params=config_params)
#configs = generate_nn_configs(config_params)
#for i in range(17):
 #   x = random.choice(configs)
  #  #print(x)
   # model2 = LinearRegression(input_size=9, output_size=1, config=x)
    #print(model2)
    #print(model2.learning_rate)
#find_best_nn(train_loader, val_loader,config_params )
#print(configs)