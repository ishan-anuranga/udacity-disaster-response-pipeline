"""
Preprocessing of Data
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)

Sample Script Syntax:
> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>

Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db

Arguments Description:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Messages Data with Categories Function
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """
    Clean Categories Data Function
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """
    
    # Split the categories
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Create category column names 
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]) 
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column])
    
    # Clean dataframe
    del df['categories']
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Filepath of the databse
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = database_filename.replace(".db","").split('/')[1] + "_table"
    print(f"table name: {table_name}")
    df.to_sql(table_name, engine, index=False)


def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
    
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