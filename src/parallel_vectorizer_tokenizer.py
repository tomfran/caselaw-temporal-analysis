from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvect
import numpy as np
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import spacy
import dask.dataframe as ddf
from collections import defaultdict

from .abstract_classes import *

pd.options.mode.chained_assignment = None  # default='warn'

def dummy(x):
    return x

class BatchTokenizer(Tokenizer):
    
    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "summary", "keywords", "article"])
    stop_words.extend(Tokenizer.too_frequent_words)
    nlp = spacy.load("en_core_web_sm")

    def remove_newlinechars(self, text):
        regex = r'\s+'
        return re.sub(regex, ' ', text)

    def nltk_tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        return list(filter(lambda word: word.isalnum(), tokens))
        

    def remove_stopwords(self, words):
        filtered = filter(lambda word: word not in BatchTokenizer.stop_words, words)    
        return list(filtered)

    def lemmatize(self, text, nlp=nlp):
        doc = nlp(" ".join(text))    
        lemmatized = [token.lemma_ for token in doc]
        return lemmatized

    def clean_text(self, df):
        df["text"] = df.text.map(lambda text:text.lower()).map(self.remove_newlinechars).map(self.nltk_tokenize).map(self.remove_stopwords)
        # .map(self.lemmatize)
        return df
    
    def tokenize(self, documents):
        df = pd.DataFrame()
        df = df.assign(text=documents)
        dask_dataframe = ddf.from_pandas(df, npartitions=4)
        result = dask_dataframe.map_partitions(self.clean_text, meta=df)
        df = result.compute()
        # df = self.clean_text(df)
        ll = df.values.tolist()
        return [el[0] for el in ll]
    
    def remove_useless(self, tokens):
        
        def check(n): 
            return n < len(tokens)*0.90
        
        freq = defaultdict(lambda : 0)
        for doc in tokens:
            for word in set(doc):
                freq[word] += 1   
            
        return [[el for el in doc if check(freq[el])] for doc in tokens]
        
class FastTfIdfVectorizer(Vectorizer):
    
    vectors_save_path="../data/processed/fast_tfidf.npy"
    vectorizer_save_path="../data/models/fast_tfidf.pickle"
    
    def __init__(self, documents, tokenizer):
        super().__init__(documents, tokenizer)
        self.vectorizer = tfidfvect(analyzer="word", 
                                    tokenizer=dummy, 
                                    preprocessor=dummy, 
                                    token_pattern=None)

    def vectors(self):
        tokens = self.tokenizer.tokenize(self.documents)
        tokens = self.tokenizer.remove_useless(tokens)
        return self.vectorizer.fit_transform(tokens)

    def vec(self, document):
        tokens = self.tokenizer.tokenize(document)
        tokens = self.tokenizer.remove_useless(tokens)
        return self.vectorizer.transform(tokens)

    def increaseWeightImportantWords(self, vectors, multiplier=2.0):
        def preProcessing(words):
            tokens = self.tokenizer.tokenize(words)
            tokens = self.tokenizer.remove_useless(tokens)
            return tokens

        for importantWord in preProcessing([word for topicWords in self.important_topics.values() for word in topicWords]):
            try:
                position = self.vectorizer.vocabulary_[importantWord[0]]
                vectors[:, position] *= multiplier
            except Exception as e:
                print(f"{e} not present")

    def save_vectors_vectorizer(self, vectors):
        with open(FastTfIdfVectorizer.vectors_save_path, "wb") as f:
            np.save(f, vectors)
        
        with open(FastTfIdfVectorizer.vectorizer_save_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
    
    @staticmethod
    def load_vectors_vectorizer():
        with open(FastTfIdfVectorizer.vectors_save_path, "rb") as f:
            loaded_vectors = np.load(f, allow_pickle=True).item()
        
        with open(FastTfIdfVectorizer.vectorizer_save_path, "rb") as f:
            loaded_vectorizer = pickle.load(f)
        
        return loaded_vectors, loaded_vectorizer