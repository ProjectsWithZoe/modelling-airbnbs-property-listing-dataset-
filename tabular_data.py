import pandas as pd
def remove_rows_with_missing_ratings(file):
    df = pd.read_csv(file)
    print(df)
    #return df

remove_rows_with_missing_ratings(file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv")
""
