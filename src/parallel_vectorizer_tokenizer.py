from sklearn.feature_extraction.text import TfidfVectorizer as tfidfvect
import numpy as np
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import spacy
import dask.dataframe as ddf

from .abstract_classes import *

pd.options.mode.chained_assignment = None  # default='warn'

def dummy(x):
    return x

class BatchTokenizer(Tokenizer):
    
    stop_words = stopwords.words("english")
    stop_words.extend(["from","subject","summary","keywords","article"])
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

class FastTfIdfVectorizer(Vectorizer):
    
    vectors_save_path="../data/processed/tfidf.npy"
    vectorizer_save_path="../data/models/tfidf.pickle"
    
    def __init__(self, documents, tokenizer):
        super().__init__(documents, tokenizer)
        self.vectorizer = tfidfvect(analyzer="word", 
                                    tokenizer=dummy, 
                                    preprocessor=dummy, 
                                    token_pattern=None)

    def vectors(self):
        tokens = self.tokenizer.tokenize(self.documents)
        return self.vectorizer.fit_transform(tokens)

    def vec(self, document):
        return self.vectorizer.transform([document])
    
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