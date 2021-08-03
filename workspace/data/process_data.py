import sys
import pandas as pd
from sqlalchemy import create_engine


"""
To run ETL pipeline that cleans data and stores in database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
"""



def load_data(messages_filepath, categories_filepath):
    """
    Loads the data to be processed

    INPUT:
    messages_filepath: file path of the message csv file 
    categories_filepath: file path of the categories csv file 

    OUTPUT:
    df: a dataframe combining the 2 csv files
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories =  pd.read_csv(categories_filepath)

# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps



    # merge datasets
    df = messages.merge(categories, on='id',how='inner')
    df.head()

    return df


def clean_data(df):
    """
    Cleans the df spliting columns, build a 0/1 columns for each category, etc
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)



    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0]for x in row]



    # rename the columns of `categories`
    categories.columns = category_colnames

    # ### 4. Convert category values to just numbers 0 or 1.

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].str.replace('2', '1')
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])


    # drop the original categories column from `df`
    df = df.drop(columns='categories')


    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)


    # drop duplicates
    df = df.drop_duplicates(subset=['id'])

    return df

def save_data(df, database_filename):
    """
    Saves the DataFrame into a database

    INPUT:
    df: dataframe
    database_filename: name of the database to be saved
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Message', engine, index=False, if_exists='replace')


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