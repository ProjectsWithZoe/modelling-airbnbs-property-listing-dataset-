import pandas as pd
import ast


file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
df = pd.read_csv(file)
def remove_rows_with_missing_ratings(df):
    
    clean_nulls = df['Cleanliness_rating'].isnull()
    accu_nulls = df['Accuracy_rating'].isnull()
    comm_nulls = df['Communication_rating'].isnull()
    location_nulls = df['Location_rating'].isnull()
    check_nulls = df['Check-in_rating'].isnull()
    value_nulls = df['Value_rating'].isnull()
    desc_null = df['Description'].isnull()
    df =(df[~clean_nulls])
    df=(df[~accu_nulls])
    df=(df[~comm_nulls])
    df=(df[~location_nulls])
    df=(df[~check_nulls])
    df=(df[~value_nulls])
    df = df[~desc_null]
    return df

new_df = remove_rows_with_missing_ratings(df)

def combine_description_strings(new_df):
    new_df['Description'] = new_df['Description'].apply(lambda x: x.replace("'About this space',",''))
    new_df['Description'] = new_df['Description'].apply(lambda x: x.replace("'",'"'))
    new_df['Description'] = new_df['Description'].apply(lambda x: x.replace('"',''))
    new_df['Description'] = new_df['Description'].apply(lambda x: x.replace(",",''))
    new_df['Description'] = new_df['Description'].apply(lambda x: "".join(x))
    return new_df

def set_default_feature_values(new_df):
    new_df[['guests', 'beds', 'bathrooms','bedrooms']] = new_df[['guests', 'beds', 'bathrooms','bedrooms']].fillna(value=1)
    print((new_df.head()))
    return new_df

set_default_feature_values(new_df)




#new_df['Description'] = new_df['Description'].str.join("\n")





