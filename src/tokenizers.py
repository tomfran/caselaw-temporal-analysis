from abc import ABC, abstractmethod
import spacy
import nltk

class Tokenizer(ABC): 
    
    @abstractmethod
    def tokenize(self, text):
        pass
    
class SpacyTokenizer(Tokenizer):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, text):
        return [token.lemma_ for token in self.nlp(text) if
                token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']]

class NLTKTokenizer(Tokenizer):

    def tokenize(self, text):
        return nltk.word_tokenize(text)
        
if __name__ == "__main__": 
    
    s = "An Apple 'fell', \tfrom the\n tree"
    t = SpacyTokenizer()
    print(t.tokenize(s))
    
    t = NLTKTokenizer()
    print(t.tokenize(s))
    