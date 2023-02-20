import pandas as pd
from ast import literal_eval
#file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
#raw_df = pd.read_csv(file)
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
    new_df['Description'] = new_df['Description'].apply(lambda x: x.replace(",",''))
    new_df['Description'] = new_df['Description'].apply(lambda x: literal_return(x))
    new_df['Description']=new_df['Description'].str.join(",")
    return new_df

def set_default_feature_values(new_df):
    new_df[['guests', 'beds', 'bathrooms','bedrooms']] = new_df[['guests', 'beds', 'bathrooms','bedrooms']].fillna(value=1)
    #print(new_df.isnull().sum())
    return new_df

def clean_tabular_data(df):
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df

def load_airbnb(label):
    cleaned_df = clean_tabular_data(raw_df)
    nums = ['float64', 'int64']
    cleaned_df = cleaned_df.select_dtypes(include=nums)
    labels = cleaned_df[label]
    features = cleaned_df.drop([label], axis =1)
    #print (features, labels)
    return (features, labels)
    

#df = combine_description_strings(new_df=remove_rows_with_missing_ratings(raw_df))
#print((df['Description'][1]))


if __name__ == '__main__':
    file="/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/listing.csv"
    raw_df = pd.read_csv(file)
    new_df = clean_tabular_data(raw_df)
    csv = pd.DataFrame(new_df).to_csv('/Users/gebruiker/modelling-airbnbs-property-listing-dataset-/airbnb-property-listings/tabular_data/clean_tabular_data.csv')




# %%
