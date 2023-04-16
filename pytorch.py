import torch
import numpy as np
import pandas as pd
import yaml
import tabular_data

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

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

'''Testing the working of the model'''
#a = AirbnbNightlyPriceRegressionDataset(data)
#features, label = a[3]
#print(label)
#print(features.shape)
#print(label.shape)
#features, labels = next(iter(train_loader))
#print(features.shape)

def get_nn_config(file):
    with open (file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        #print(data)
        return data
config_path = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/nn_config.yaml'

class LinearRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, config):
        super(LinearRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.optimiser = config['optimiser']
        self.learning_rate = config['learning_rate']
        self.hidden_layer_width = config['hidden_layer_width']
        self.model_depth = config['model_depth']

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
input_size = 9
hidden_size = 3
output_size = 1

model = LinearRegression(input_size, hidden_size, output_size, config=get_nn_config(config_path))

def train(model, train_loader, val_loader, num_epochs=10):
    if model.optimiser == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=model.learning_rate)
    else:
        raise ValueError ('Optimiser not supported')
    writer = SummaryWriter()

    #batch_idx = 0
    for epoch in range(num_epochs):
        #model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            features, label = batch
            predictions = model(features)
            loss = F.mse_loss(predictions, label)
            train_loss +=loss.item()
            
            #print(loss.item())
            #print(model.parameters())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            
            writer.add_scalar('loss', loss.item(), epoch*len(train_loader)+i)
        train_loss /= len(train_loader)
        writer.add_scalar('avg train loss', train_loss, epoch)
            #print(features.shape) # Ensure output shape is correct
            #break # Only get first batch of data
        val_loss = 0.0
        with torch.no_grad():
            for j, val_batch in enumerate(val_loader):
                val_features, val_label = val_batch
                val_predictions = model(val_features)
                val_loss += F.mse_loss(val_predictions, val_label).item()
            val_loss /=len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss}, Val loss: {val_loss}')
        writer.add_scalar('val loss', loss.item(), epoch*len(val_loader)+i)
        #writer.add_scalar('avg val loss', val_loss, epoch)


train(model, train_loader, val_loader)
sd = model.state_dict()
torch.save(model.state_dict(), 'model.pt')


#get_nn_config(file='/nn_config.yaml')
