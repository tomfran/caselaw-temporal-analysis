import os
import spacy
import json
from string import punctuation, digits

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        
class BatchTokenizer():
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
    
    def tokenize(self, documents):
        tokens = []
        for doc in self.nlp.pipe(documents, n_process=-1):
            tokens.append([token.lemma_.lower() for token in doc 
                        if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']])
            
        return [[x for x in doc if x not in punctuation and x not in digits] 
                for doc in tokens]
    
    def save_tokens(self, 
                    tokens, 
                    tokens_save_path="../data/processed/tokens.json"):
        with open(tokens_save_path, "w") as f:
            f.write(json.dumps(tokens))    
            
    @staticmethod
    def load_tokens(tokens_save_path="../data/processed/tokens.json"):
        with open(tokens_save_path, "r") as f:
            return json.load(f)