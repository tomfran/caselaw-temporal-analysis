from abc import ABC, abstractmethod
from tokenizers import *
from sklearn.feature_extraction.text import TfidfVectorizer

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
    
class TfIdfVectors(Vectorizer):

    def __init__(self, documents, tokenizer):
        super().__init__(documents, tokenizer)
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer.tokenize)

    def vectors(self):
        return self.vectorizer.fit_transform(self.documents)

    def vec(self, document):
        return self.vectorizer.transform([document])
    
if __name__ == "__main__":
    doc = ["An apple fell from the tree", 
           "apple apple apple apple apple was founded in 4543 inside a tree"]
    
    t = SpacyTokenizer()
    v = TfIdfVectors(doc, t)
    
    from pprint import pprint
    print(v.vectors())
    print(v.vectorizer.vocabulary_)