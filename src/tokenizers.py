from abc import ABC, abstractmethod
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from functools import reduce

import spacy
import nltk

from .abstract_classes import Tokenizer


class DocumentTokenizer(Tokenizer):

    def __init__(self, n_workers=4):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize_sentences(self, text):        
        return nltk.sent_tokenize(text)
    
    def spacy_tokenization(self, text):
        return [token.lemma_ for token in self.nlp(text) if
                token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']]
        
    def tokenize(self, text, get_sentences=True):
        if get_sentences:
            sentences = self.tokenize_sentences(text)            
            tokens = []
            for sent in self.nlp.pipe(sentences, n_process=-1):
                tokens += [token.lemma_ for token in sent 
                           if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']]
            return tokens
        else:
            return self.spacy_tokenization(text)
        
class BatchTokenizer(Tokenizer):
    
    def __init__(self, n_workers=4):
        self.nlp = spacy.load("en_core_web_sm")
    
    def tokenize(self, documents):
        tokens = []
        for doc in self.nlp.pipe(documents, n_process=-1):
            tokens.append([token.lemma_ for token in doc 
                        if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']])
        return tokens
    
if __name__ == "__main__":
    
    from pprint import pprint
    import timeit
    
    t = DocumentTokenizer()
    
    text = "".join(open("text.txt", "r").readlines())
    
    print(timeit.timeit('t.tokenize(text)',
                        setup='from __main__ import t, text', 
                        number=1))
    print(timeit.timeit('t.tokenize(text, False)',
                        setup='from __main__ import t, text', 
                        number=1))