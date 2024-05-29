from textblob import TextBlob

class Sentiment:
    def __init__(self):
        self.__text = None
        self.__sentiment_score = None

    def analyze_sentiment(self, text: str):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def categories_sentiment(self, sentiment_score: float):
        # Check polarity for sentiment (-1 to 1, negative to positive)
        if sentiment_score > 0:
            return "Positive"
        elif sentiment_score == 0:
            return "Neutral"
        else:
            return "Negative"