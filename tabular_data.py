import pandas as pd


file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
df = pd.read_csv(file)
def remove_rows_with_missing_ratings(df):
    
    clean_nulls = df['Cleanliness_rating'].isnull()
    accu_nulls = df['Accuracy_rating'].isnull()
    comm_nulls = df['Communication_rating'].isnull()
    location_nulls = df['Location_rating'].isnull()
    check_nulls = df['Check-in_rating'].isnull()
    value_nulls = df['Value_rating'].isnull()
    df =(df[~clean_nulls])
    df=(df[~accu_nulls])
    df=(df[~comm_nulls])
    df=(df[~location_nulls])
    df=(df[~check_nulls])
    df=(df[~value_nulls])
    return df

new_df = remove_rows_with_missing_ratings(df)
print(new_df)

