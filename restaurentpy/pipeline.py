from restaurentpy.topic_model import TopicModel
class RunPipeline:
    def __init__(self, path: str, pat: str):
        self.topic = TopicModel(path, pat)

    def run_pipeline(self):
        df = self.topic.get_topics()
        
        return df        
        
        
        
        