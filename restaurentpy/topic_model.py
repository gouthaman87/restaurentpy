from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

class TopicModel:
    def __init__(self):
        pass
    
    def user_defined_topic_model(self, rev_number: list, review_dates: list, documents: list, model: str, topic: list):
        # Load a pre-trained BERT model for embedding the documents
        model = SentenceTransformer(model)
        
        # compute embeddings for the user-defined topics
        topic_embeddings = model.encode(topic)

        # Create a dictionary to store the topic embeddings
        topic_embedding_dict = {topic: emb for topic, emb in zip(topic, topic_embeddings)}
        
        # Generate embeddings for the documents using the pre-trained model
        embeddings = model.encode(documents)
        
        # assign topics based on user-defined embeddings
        custom_topics = self.assign_custom_topics(embeddings, topic_embedding_dict)
        
        df_topic = pd.DataFrame({
            'review_id': rev_number,
            'calendar_date': review_dates,
            'Topic': custom_topics
        })
        
        return df_topic
        
    def assign_custom_topics(self, doc_embeddings, topic_embeddings_dict):
        assigned_topics = []
        for doc_emb in doc_embeddings:
            # Calculate cosine similarity with each user-defined topic
            similarities = {topic: np.dot(doc_emb, topic_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(topic_emb))
                            for topic, topic_emb in topic_embeddings_dict.items()}
            # Assign the topic with the highest similarity
            assigned_topic = max(similarities, key=similarities.get)
            assigned_topics.append(assigned_topic)
        return assigned_topics