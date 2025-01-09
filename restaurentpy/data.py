import pandas as pd
import os
import logging

class ReviewData:
    def __init__(self, path: str, pat: str) -> None:
        """"
        Initializes to read review data
        
        Args:
            path (str): The file path where data saved
            pat (str) : The type of file to read (e.g. xlsx)
        """
        self.path = path
        self.pat = pat
        self.df = None
        
    def read_review(self):
        """
        The function to read review files
        
        Returns:
            DataFrame: The pandas DataFrame (Review DataFrame)
        """
        # List to store the data frames
        self.df = []
        
        # Get all the files in the folder
        files = os.listdir(self.path)
        
        for file in files:
            if file.endswith(self.pat): # Filter files based on the pat
                file_path = os.path.join(self.path, file)
                logging.info("Reading File: ", file_path)
                self.df.append(pd.read_excel(file_path))
                
        # Concatenate DataFrames along rows
        self.df = pd.concat(self.df, axis=0, ignore_index=True)
        
        return self.df
    
    def etl_review(self):
        """
        The function to do ETL on review data

        Returns:
            DataFrame: The pandas DataFrame
        """
        df = self.read_review()
        
        # Select columns need for Analysis
        df = df.loc[:,['name', 'review_id', "review_datetime_utc", "review_text", "review_rating"]]
        df['branch'] = df['name']
        
        # Get Month year from datetime column
        df['calendar_date'] = pd.to_datetime(df['review_datetime_utc']).apply(lambda x: x.strftime('%B-%Y')) 
        df = df[['review_id', 'branch', "calendar_date", "review_text", "review_rating"]]
        
        # Drop Duplicates
        df = df.drop_duplicates()
        
        return df