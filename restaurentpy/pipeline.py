from restaurentpy.data import ReviewData
from restaurentpy.translate import ReviewTranslate
from restaurentpy.sentiment import Sentiment
from restaurentpy.topic_model import TopicModel
import logging
from cleantext import clean
import pandas as pd

class RunPipeline:
    def __init__(self, path: str, pat: str, model: str, topic: list):
        self.data = ReviewData(path=path, pat=pat)
        self.translate = ReviewTranslate()
        self.sentiment = Sentiment()
        self.topic_model = TopicModel()
        self.model = model
        self.topic = topic

    def run_pipeline(self):
        """Running pipeline for topic model
        - Cleaning reviews.
        - Calculate sentiment score.
        - Assign user defined topics.

        Returns:
            Data Frame: The final data frame with sentiment & topics for reviews
        """
        # 01. Read Data ----
        print(f"STEP 01: Reading Data ...")
        df_raw = self.data.etl_review()
        
        # Get the number of rows
        print(f"Number of rows of Raw Data: {df_raw.shape[0]}")
        
        # 02. Identify / Detect the language of the text ----
        print("STEP 02: Identify language for reviews ...")
        df_etl = df_raw.copy()
        df_etl['lang'] = df_etl.review_text.map(lambda x: self.translate.get_language(x))
        
        # 03. Translate non english reviews ----
        print("STEP 03: Translate Reviews ...")
        df_etl['translate_review'] = df_etl.apply(lambda row: self.translate.translate(row['lang'], row['review_text']), axis=1)
        
        # 04. Remove shorter reviews ----
        print("STEP 04: Remove Shorter Reviews ...")
        df_etl = df_etl.loc[df_etl['translate_review'].str.len() > 8]
        print(f'Number of Reviews after cleaning: {df_etl.shape[0]}')
        
        # 05. Cleaning emojis ----
        print("STEP 05: Cleaning emojis in reviews ...")
        df_etl['translate_review'] = df_etl['translate_review'].apply(lambda x: clean(x, no_emoji=True))
        
        # 06. Calculate Sentiment ----
        print("STEP 06: Calculate sentiment scores ...")
        df_etl['sentiment_score'] = df_etl['translate_review'].apply(self.sentiment.analyze_sentiment)
        df_etl['sentiment_type'] = df_etl['sentiment_score'].apply(self.sentiment.categories_sentiment)
        df_etl = df_etl.drop(columns=['branch', 'review_text', 'review_rating'])
        
        # 07. Fit Defined Topics Model (embedding algorithm) ----
        print("STEP 07: Find topics in reviews ...")
        documents = list(df_etl.translate_review.values)
        review_number = list(df_etl.review_id.values)
        calendar_dates = list(df_etl.calendar_date.values)
        df_topic = self.topic_model.user_defined_topic_model(rev_number=review_number,
                                                             review_dates=calendar_dates,
                                                             documents=documents, 
                                                             model=self.model, 
                                                             topic=self.topic)
        
        # Get the number of rows
        print(f"Number of rows Topic Data: {df_topic.shape[0]}")
        
        # 08. Create Final Data ----
        print(f"STEP 08: Creating Final Data ----")
        df_final = pd.merge(df_raw, df_etl, on=['review_id', 'calendar_date'], how='left')
        df_final = pd.merge(df_final, df_topic, on=['review_id', 'calendar_date'], how='left')
       
        # Get the number of rows
        print(f"Number of rows Final Data: {df_final.shape[0]}")

        return df_final