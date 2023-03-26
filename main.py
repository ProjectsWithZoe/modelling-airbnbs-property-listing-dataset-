import tabular_data, modelling
import pandas as pd
file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)
df = tabular_data.clean_tabular_data(raw_df)

labels, features = tabular_data.load_airbnb(df, 'Category')
print(labels.unique())

