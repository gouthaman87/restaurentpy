import langdetect # Package to identify languages
from deep_translator import GoogleTranslator

class ReviewTranslate:
    def __init__(self):
        """
        Initialize the translation module
        """
        self.__text = None
        self.__language = None
    
    def get_language(self, text: str):
        """
        The function to detect which language the reviews
        
        Args:
            text (str): The reviews

        Returns:
            str: The detected language
        """
        try:
            return langdetect.detect(text)
        except KeyboardInterrupt as e:
            raise(e)
        except:
            return '<-- ERROR -->'
        
    def translate(self, language: str, text: str):
        """
        The function to google translate if reviews are not english
        
        Args:
            language (str): The language of the review written
            text (str): The reviews

        Returns:
            str: The translated reviews
        """
        if language != "en":
            text = GoogleTranslator(source='auto', target='en').translate(str(text))
        else:
            text
        return text
        