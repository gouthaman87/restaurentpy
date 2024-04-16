import pandas as pd
import os

class ReviewData:
    def __init__(self, path: str) -> None:
        '''
        Initializes to read review data
        
        Args:
            path (str): The file path where data saved
        '''
        self.path = path
        
        
    def read_review(self, pat: str):
        '''
        The function to read files
        
        Args:
            pat (str): The type of file to read (e.g. xlsx)
        '''
        if pat in ['xlsx']:
            files = os.listdir(self.path)
            
            # print("Reading From:", self.path)
            # df = pd.read_excel(self.path)
        return files
        