import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function load message and categories data and return the merged dataframe 
    The dataset are in csv format pandas read_csv method is used to read data from given location
    
    
    Args/Input :
        messages_filepath : The path of message dataset
        categories_filepath: The path of caregories dataset
        
    Output :
        The merged data (messgae, Categories) are returend as dataframe 
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df

    
    


def clean_data(df):
    """
    This  function perform cleaning of data operation. 
    select only required data
    Remove duplicate row
    
    Args/Input : 
    Megred df from load_data(args) method
    
    Output :
    returned cleaned dataframe 
    
    """
    
    #splitting categories 
    categories = df.categories.str.split(';',expand=True)
    row = categories.loc[0]
    #setting each category in different column
    category_names = row.apply(lambda x:x[:-2])
    categories.columns = category_names
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    
    df.drop('categories',axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates(subset='id')
    return df
        
    
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disaster',engine,index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()