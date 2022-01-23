import os
import spacy
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        
class BatchTokenizer():
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
    
    def tokenize(self, documents):
        tokens = []
        for doc in self.nlp.pipe(documents, batch_size = 30):
            tokens.append([token.lemma_ for token in doc 
                           if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']])
            
        return tokens