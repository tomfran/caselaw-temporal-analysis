from abc import ABC, abstractmethod
import spacy
import nltk
from functools import reduce

from .abstract_classes import Tokenizer

class DocumentTokenizer(Tokenizer):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize_sentences(self, text):        
        return nltk.sent_tokenize(text)
    
    def spacy_tokenization(self, text):
        return [token.lemma_ for token in self.nlp(text) if
                token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']]
        
    def tokenize(self, text, get_sentences=False):
        if get_sentences:
            sentences = self.tokenize_sentences(text)
            return reduce(lambda a, b: a + b,
                          [self.spacy_tokenization(sent) for sent in sentences])
        else:
            return self.spacy_tokenization(text)
        
if __name__ == "__main__":
    
    from pprint import pprint
    import timeit
    
    t = DocumentTokenizer()
    
    text = "".join(open("text.txt", "r").readlines())
    
    print(timeit.timeit('t.tokenize(text)',
                        setup='from __main__ import t, text', 
                        number=3))