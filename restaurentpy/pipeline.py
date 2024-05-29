from restaurentpy.data import ReviewData
from restaurentpy.translate import ReviewTranslate
from restaurentpy.sentiment import Sentiment
import logging
from cleantext import clean

class RunPipeline:
    def __init__(self, path: str, pat: str):
        self.data = ReviewData(path=path, pat=pat)
        self.translate = ReviewTranslate()
        self.sentiment = Sentiment()

    def run_pipeline(self):
        # Read Data
        df = self.data.etl_review()
        
        # Identify / Detect the language of the text
        logging.info("Identify language for reviews ...")
        df['lang'] = df.review_text.map(lambda x: self.translate.get_language(x))
        
        # Extract English & Danish Reviews
        # TODO: Later we must use all the reviews and then need to translate
        filtered_indices = df['lang'].str.contains('en|de|ar')
        df = df.loc[filtered_indices, ]
        
        # Translate non english reviews
        logging.info("Translate Reviews ...")
        df['translate_review'] = df.apply(lambda row: self.translate.translate(row['lang'], row['review_text']), axis=1)
        
        # Remove shorter reviews
        logging.info("Remove Shorter Reviews ...")
        df = df.loc[df['translate_review'].str.len() > 8]
        logging.info(f'Number of Reviews: {df.shape[0]}')
        
        # Cleaning emojis
        logging.info("Cleaning emojis in reviews ...")
        df['translate_review'] = df['translate_review'].apply(lambda x: clean(x, no_emoji=True))
        
        # Calculate Sentiment
        df['sentiment_score'] = df['translate_review'].apply(self.sentiment.analyze_sentiment)
        df['sentiment_type'] = df['sentiment_score'].apply(self.sentiment.categories_sentiment)

        return df 
        
        
        
        