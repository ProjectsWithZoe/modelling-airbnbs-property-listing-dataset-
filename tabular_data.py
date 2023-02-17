import pandas as pd
from ast import literal_eval
file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
raw_df = pd.read_csv(file)
def remove_rows_with_missing_ratings(df):
    df = df.dropna(subset=['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating', 'Check-in_rating', 'Value_rating'])
    return df

def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val


def combine_description_strings(new_df):
    
    new_df = new_df.dropna(subset=['Description'])
    new_df = new_df.copy()
    new_df['Description'] = new_df['Description'].apply(lambda x: x.replace("'About this space',",''))
    #new_df['Description'] = new_df['Description'].apply(lambda x: x.replace("'",'"'))
    #new_df['Description'] = new_df['Description'].apply(lambda x: x.replace('"',''))
    #new_df['Description'] = new_df['Description'].apply(lambda x: x.replace(",",''))
    #new_df['Description'] = new_df['Description'].apply(lambda x: "".join(x))
    new_df['Description'] = new_df['Description'].apply(lambda x: literal_return(x))
    return new_df
    
    #print(new_df.head())
    #return new_df

def set_default_feature_values(new_df):
    new_df[['guests', 'beds', 'bathrooms','bedrooms']] = new_df[['guests', 'beds', 'bathrooms','bedrooms']].fillna(value=1)
    print(new_df.isnull().sum())
    #print((new_df.head()))
    return new_df

def clean_tabular_data(df):
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df

df = combine_description_strings(new_df=remove_rows_with_missing_ratings(raw_df))
print(type(df['Description'][0]))


if __name__ == '__main__':
    file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
    raw_df = pd.read_csv(file)
    new_df = clean_tabular_data(raw_df)
    csv = pd.DataFrame(new_df).to_csv('/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/clean_tabular_data.csv')




# %%
