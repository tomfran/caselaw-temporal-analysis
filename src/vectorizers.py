from abc import ABC, abstractmethod
from .tokenizers import *
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvect, CountVectorizer as countvec
import numpy as np
import pickle

class Vectorizer(ABC):

    def __init__(self, documents, tokenizer):
        self.tokenizer = tokenizer
        self.documents = documents

    @abstractmethod
    def vectors(self):
        pass

    @abstractmethod
    def vec(self, document):
        pass
    
class TfIdfVectorizer(Vectorizer):
    
    vectors_save_path="../data/processed/tfidf.npy"
    vectorizer_save_path="../data/models/tfidf.pickle"
    
    def __init__(self, documents, tokenizer):
        super().__init__(documents, tokenizer)
        self.vectorizer = tfidfvect(tokenizer=self.tokenizer.tokenize)

    def vectors(self):
        return self.vectorizer.fit_transform(self.documents)

    def vec(self, document):
        return self.vectorizer.transform([document])
    
    def save_vectors_vectorizer(self, vectors):
        with open(TfIdfVectorizer.vectors_save_path, "wb") as f:
            np.save(f, vectors)
        
        with open(TfIdfVectorizer.vectorizer_save_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
    
    @staticmethod
    def load_vectors_vectorizer():
        with open(TfIdfVectorizer.vectors_save_path, "rb") as f:
            loaded_vectors = np.load(f, allow_pickle=True).item()
        
        with open(TfIdfVectorizer.vectorizer_save_path, "rb") as f:
            loaded_vectorizer = pickle.load(f)
        
        return loaded_vectors, loaded_vectorizer

class MyCountVectorizer(Vectorizer):

    vectors_save_path="../data/processed/count.npy"
    vectorizer_save_path="../data/models/count.pickle"

    def __init__(self, documents, tokenizer):
        super().__init__(documents, tokenizer)
        self.vectorizer = countvec(tokenizer=self.tokenizer.tokenize)

    def vectors(self):
        return self.vectorizer.fit_transform(self.documents)

    def vec(self, document):
        return self.vectorizer.transform([document])

    def save_vectors_vectorizer(self, vectors):
        with open(MyCountVectorizer.vectors_save_path, "wb") as f:
            np.save(f, vectors)

        with open(MyCountVectorizer.vectorizer_save_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @staticmethod
    def load_vectors_vectorizer():
        with open(MyCountVectorizer.vectors_save_path, "rb") as f:
            loaded_vectors = np.load(f, allow_pickle=True).item()

        with open(MyCountVectorizer.vectorizer_save_path, "rb") as f:
            loaded_vectorizer = pickle.load(f)

        return loaded_vectors, loaded_vectorizer
    
    
if __name__ == "__main__":
    doc = ["An apple fell from the tree", 
           "apple apple apple apple apple was founded in 4543 inside a tree"]
    
    t = SpacyTokenizer()
    v = TfIdfVectors(doc, t)
    
    from pprint import pprint
    print(v.vectors())
    print(v.vectorizer.vocabulary_)