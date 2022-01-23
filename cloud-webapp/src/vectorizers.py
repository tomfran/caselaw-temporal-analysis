import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def identity_tokenizer(x):
    return x

class TokenVectorizer():
    
    def __init__(self, documents, method="tfidf"):
        
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, 
                                              lowercase=False)
        else:
            self.vectorizer = CountVectorizer(tokenizer=identity_tokenizer, 
                                              lowercase=False)
        self.method = method
        
        self.documents = documents
        
    def vectors(self):
        return self.vectorizer.fit_transform(self.documents)

    def vec(self, document):
        return self.vectorizer.transform([document])
    
    def save_vectors_vectorizer(self, vectors):
        
        vectors_save_path = f"../data/processed/{self.method}.npy"
        vectorizer_save_path = f"../data/models/{self.method}.pickle"
        np.save(open(vectors_save_path, "wb"), 
                vectors)
        
        pickle.dump(self.vectorizer, 
                    open(vectorizer_save_path, "wb"))
    
    @staticmethod
    def load_vectors_vectorizer(method="tfidf"):
        
        vectors_save_path = f"../data/processed/{method}.npy"
        vectorizer_save_path = f"../data/models/{method}.pickle"
        
        loaded_vectors = np.load(open(vectors_save_path, "rb"), 
                                 allow_pickle=True).item()
    
        loaded_vectorizer = pickle.load(open(vectorizer_save_path, "rb"))    
        
        return loaded_vectors, loaded_vectorizer