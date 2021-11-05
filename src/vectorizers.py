from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvect
import pickle
import numpy as np

from .tokenizers import DocumentTokenizer
from .abstract_classes import Vectorizer

class DocumentVectorizer(Vectorizer):
    
    def __init__(self, documents, tokenizer, min_max_df=(1, 1.0)):
        super().__init__(documents, tokenizer)
        self.vectorizer = tfidfvect(tokenizer=self.tokenizer.tokenize, 
                                    max_df=min_max_df[1], 
                                    min_df=min_max_df[0])
        
    def vectors(self):
        return self.vectorizer.fit_transform(self.documents)

    def vec(self, document):
        return self.vectorizer.transform([document])
    
    def save_vectors_vectorizer(self, 
                                vectors, 
                                vectors_save_path="../data/processed/tfidf.npy", 
                                vectorizer_save_path="../data/models/tfidf.pickle"):
        
        np.save(open(vectors_save_path, "wb"), 
                vectors)
        
        pickle.dump(self.vectorizer, 
                    open(vectorizer_save_path, "wb"))
    
    @staticmethod
    def load_vectors_vectorizer(vectors_save_path="../data/processed/tfidf.npy", 
                                vectorizer_save_path="../data/models/tfidf.pickle"):
        
        loaded_vectors = np.load(open(vectors_save_path, "rb"), 
                                 allow_pickle=True).item()
    
        loaded_vectorizer = pickle.load(open(vectorizer_save_path, "rb"))    
        
        return loaded_vectors, loaded_vectorizer

