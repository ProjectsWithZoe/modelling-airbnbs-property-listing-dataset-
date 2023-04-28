import pytorch
import pandas as pd
import tabular_data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)
data= tabular_data.clean_tabular_data(df=raw_df)
nums = ['float64', 'int64']
data = data.select_dtypes(include = nums)
label_column = 'beds'

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_dataset = pytorch.AirbnbNightlyPriceRegressionDataset(train_data, label_column)
val_dataset = pytorch.AirbnbNightlyPriceRegressionDataset(val_data, label_column)
test_dataset = pytorch.AirbnbNightlyPriceRegressionDataset(test_data, label_column)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

config_params = pytorch.config_params
save_dir = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/bed_model_nn/'
best_save_dir = '/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/bed_model_nn/best_nn'
print(pytorch.find_best_nn(train_loader, val_loader, config_params, save_dir, best_save_dir))
