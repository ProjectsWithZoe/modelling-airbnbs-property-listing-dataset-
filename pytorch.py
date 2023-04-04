import torch
import numpy as np
import pandas as pd
import tabular_data
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)
data= tabular_data.clean_tabular_data(df=raw_df)
nums = ['float64', 'int64']
data = data.select_dtypes(include = nums)

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self,data):
        self.label = torch.tensor(data['Price_Night'].values)
        self.features = torch.tensor(data.drop('Price_Night', axis=1).values)
    def __getitem__(self, index):
        return self.features[index], self.label[index]
    def __len__(self):
        return len(self.features)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_dataset = AirbnbNightlyPriceRegressionDataset(train_data)
val_dataset = AirbnbNightlyPriceRegressionDataset(val_data)
test_dataset = AirbnbNightlyPriceRegressionDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

#a = AirbnbNightlyPriceRegressionDataset(data)
#index3 = a.__getitem__(1)
#print (index3)


