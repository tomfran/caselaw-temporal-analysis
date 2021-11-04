from abc import ABC, abstractmethod

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
    

class Tokenizer(ABC): 
        
    @abstractmethod
    def tokenize(self, text):
        pass
    