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
        # 01. Read Data
        print(f"Reading Data ...")
        df_raw = self.data.etl_review()
        
        # Get the number of rows
        num_rows = df_raw.shape[0]
        print(f"Number of rows of Raw Data: {num_rows}")
        
        # 02. Identify / Detect the language of the text
        # logging.info("Identify language for reviews ...")
        print("Identify language for reviews ...")
        df_etl = df_raw.copy()
        df_etl['lang'] = df_etl.review_text.map(lambda x: self.translate.get_language(x))
        
        # # Extract English, Arabic & Danish Reviews
        # # TODO: Later we must use all the reviews and then need to translate
        # filtered_indices = df['lang'].str.contains('en|de|ar')
        # df = df.loc[filtered_indices, ]
        
        # 03. Translate non english reviews
        # logging.info("Translate Reviews ...")
        print("Translate Reviews ...")
        df_etl['translate_review'] = df_etl.apply(lambda row: self.translate.translate(row['lang'], row['review_text']), axis=1)
        
        # 04. Remove shorter reviews
        # logging.info("Remove Shorter Reviews ...")
        print("Remove Shorter Reviews ...")
        df_etl = df_etl.loc[df_etl['translate_review'].str.len() > 8]
        # logging.info(f'Number of Reviews after cleaning: {df.shape[0]}')
        print(f'Number of Reviews after cleaning: {df_etl.shape[0]}')
        
        # 05. Cleaning emojis
        # logging.info("Cleaning emojis in reviews ...")
        print("Cleaning emojis in reviews ...")
        df_etl['translate_review'] = df_etl['translate_review'].apply(lambda x: clean(x, no_emoji=True))
        
        # 06. Calculate Sentiment
        print("Calculate sentiment scores ...")
        df_etl['sentiment_score'] = df_etl['translate_review'].apply(self.sentiment.analyze_sentiment)
        df_etl['sentiment_type'] = df_etl['sentiment_score'].apply(self.sentiment.categories_sentiment)
        df_etl = df_etl.drop(columns=['branch', 'calendar_date', 'review_text', 'review_rating'])
        
        # 07. Fit Defined Topics Model (embedding algorithm)
        print("Find topics in reviews ...")
        documents = list(df_etl.translate_review.values)
        review_number = list(df_etl.review_number.values)
        df_topic = self.topic_model.user_defined_topic_model(rev_number=review_number,
                                                             documents=documents, 
                                                             model=self.model, 
                                                             topic=self.topic)
        
        # Get the number of rows
        num_rows = df_topic.shape[0]
        print(f"Number of rows Topic Data: {num_rows}")
        
        # 08. Create Final Data ----
        df_final = pd.merge(df_raw, df_etl, on='review_number', how='left')
        df_final = pd.merge(df_final, df_topic, on='review_number', how='left')
        df_final = df_final.drop('review_number', axis=1)
        
        # Get the number of rows
        num_rows = df_final.shape[0]
        print(f"Number of rows Final Data: {num_rows}")

        return df_final
        