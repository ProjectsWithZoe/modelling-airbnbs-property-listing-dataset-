import tabular_data
import pandas as pd
file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)

df= tabular_data.clean_tabular_data(df=raw_df)
tabular_data.load_airbnb(df,label='Price_Night')