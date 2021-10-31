from abc import ABC, abstractmethod
from .tokenizers import Tokenizer

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