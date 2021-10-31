from abc import ABC, abstractmethod

class Tokenizer(ABC): 
    
    @abstractmethod
    def tokenize(self, text):
        pass