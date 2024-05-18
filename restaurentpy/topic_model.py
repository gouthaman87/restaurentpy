from restaurentpy.data import ReviewData
from restaurentpy.translate import ReviewTranslate
import logging
from cleantext import clean

class TopicModel:
    def __init__(self, path: str, pat: str):
        self.data = ReviewData(path=path, pat=pat)
        self.translate = ReviewTranslate()
    
    def get_topics(self):
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
            
        return df