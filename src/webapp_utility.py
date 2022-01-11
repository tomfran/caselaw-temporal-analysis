import numpy as np
import pickle
import json
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from .word2vec_utils import load_models, align_models, get_similarity_sequence_base, get_similarity_sequence_consecutive
from .lda_utils import get_word_relevance, get_words_relevance, get_relevant_words

class Loader():
    
    def __init__(self, 
                 vectors_save_path_big="../data/webapp/count_big.npy",
                 vectorizer_save_path_big="../data/webapp/count_big.pickle",
                 vectors_save_path_lda_big="../data/webapp/count_lda_big.npy",
                 vectorizer_save_path_lda_big="../data/webapp/count_lda_big.pickle", 
                 vectors_save_path_lda_small="../data/webapp/count_lda_small.npy",
                 vectorizer_save_path_lda_small="../data/webapp/count_lda_small.pickle", 
                 doc_dates_topics_path="../data/webapp/doc_dates_topics.json",
                 lda_model_big_path="../data/webapp/lda_model_big.pk",
                 lda_model_small_path="../data/webapp/lda_model_small.pk", 
                 we_full_path="../data/webapp/we_full.model", 
                 we_one_year_path="../data/webapp/we_one_year_vectors", 
                 we_ten_year_path="../data/webapp/we_ten_year_vectors"):
        
        print("Loading full count vectorizers... ", end="")
        # full count distributions
        self.vectors_big = np.load(open(vectors_save_path_big, "rb"), allow_pickle=True).item()
        self.vectorizer_big = pickle.load(open(vectorizer_save_path_big, "rb"))
        self.vocab_big = self.vectorizer_big.get_feature_names()
        self.word2id_big = dict((v, idx) for idx, v in enumerate(self.vocab_big))
        self.id2word_big = dict((idx, v) for idx, v in enumerate(self.vocab_big))
        self.vectors_big_trans = self.vectors_big.transpose()
        self.doc_dates_topics = json.load(open(doc_dates_topics_path))
        
        print("Done\nLoading full lda model... ", end="")
        # big lda
        self.vectors_lda_big = np.load(open(vectors_save_path_lda_big, "rb"), allow_pickle=True).item()
        self.vectorizer_lda_big = pickle.load(open(vectorizer_save_path_lda_big, "rb"))
        self.vocab_lda_big = self.vectorizer_lda_big.get_feature_names()
        self.word2id_lda_big = dict((v, idx) for idx, v in enumerate(self.vocab_lda_big))
        self.id2word_lda_big = dict((idx, v) for idx, v in enumerate(self.vocab_lda_big))
        self.lda_model_big = pickle.load(open(lda_model_big_path, "rb"))
        
        print("Done\nLoading small lda model... ", end="")
        # small lda
        self.vectors_lda_small = np.load(open(vectors_save_path_lda_small, "rb"), allow_pickle=True).item()
        self.vectorizer_lda_small = pickle.load(open(vectorizer_save_path_lda_small, "rb"))
        self.vocab_lda_small = self.vectorizer_lda_small.get_feature_names()
        self.word2id_lda_small = dict((v, idx) for idx, v in enumerate(self.vocab_lda_small))
        self.id2word_lda_small = dict((idx, v) for idx, v in enumerate(self.vocab_lda_small))
        self.lda_model_small = pickle.load(open(lda_model_small_path, "rb"))
        
        # self.lda_small
        print("Done\nLoading word embeddings... ", end="")
        self.we_full = Word2Vec.load(we_full_path)
        self.we_one_year = load_models(we_one_year_path)
        align_models(self.we_one_year)
        self.we_ten_year = load_models(we_ten_year_path)
        align_models(self.we_ten_year)
        print("Done")
        
    def get_freq_distribution(self, words, interval=1):
        
        def _frequency_intersection(l1, l2):
            return [l1[i] and l2[i] for i in range(len(l1))]
        
        def _get_docs(word):
            ind = self.word2id_big.get(word, -1)
            return [min(1, occ) for occ in self.vectors_big_trans[ind].toarray()[0]]
        
        norm_dates = [e["decision_date"] - e["decision_date"]%interval 
                      for e in self.doc_dates_topics]

        dates_frequencies = defaultdict(lambda:0)

        for d in norm_dates:
            dates_frequencies[d] += 1

        freq_intersection = _get_docs(words[0])
        
        for word in words[1:]:
            freq_intersection = _frequency_intersection(freq_intersection, _get_docs(word))
        
        dates = [norm_dates[index] for index, occ in 
                enumerate(freq_intersection) if occ > 0]
        
        freqs = defaultdict(lambda:0)
        for year in dates:
            freqs[year] += 1

        return sorted([(year, occ/dates_frequencies[year]) 
                       for year, occ in freqs.items()])
        
    def get_n_similar(self, word, n=5, model_type="full", year=2000):
        # get top n similar words wrt the word and model_type
        try:
            if model_type == "full":
                return self.we_full.wv.most_similar(word, topn=n)
            elif model_type == "one":
                return self.we_one_year[year].wv.most_similar(word, topn=n)
            elif model_type == "ten":
                return self.we_ten_year[year].wv.most_similar(word, topn=n)
        except:
            return []
            
    def get_topic_dist(self, words, model="big"):
        
        if model == "big":
            return get_words_relevance(words, self.word2id_lda_big, 
                                       self.vocab_lda_big, self.lda_model_big, 
                                       normalize=True, precision=3)
        else:
            return get_words_relevance(words, self.word2id_lda_small, 
                                       self.vocab_lda_small, self.lda_model_small, 
                                       normalize=True, precision=3)
    
    def get_topics_words(self, n, model="big"):
        
        if model == "big":
            return get_relevant_words(self.lda_model_big, self.vocab_lda_big, n)
        else:
            return get_relevant_words(self.lda_model_small, self.vocab_lda_small, n)
    
    def get_semantic_data(self, word, model="one", base_year=2010):
        y1 = sorted(self.we_one_year.keys(), reverse=True)
        y2 = sorted(self.we_ten_year.keys(), reverse=True)
        return {"one_year"  : list(zip(y1, get_similarity_sequence_base(self.we_one_year, base_year, word))), 
                "ten_year" : list(zip(y2, get_similarity_sequence_consecutive(self.we_ten_year, word)))}
    
    def get_topics_description(self, topic=None):
        return f"Topic {topic} description"
    
    def get_topics_date_distribution(self, interval=1):
        
        norm_dates = [e["decision_date"] - e["decision_date"]%interval 
                      for e in self.doc_dates_topics]

        dates_frequencies = defaultdict(lambda:0)

        for d in norm_dates:
            dates_frequencies[d] += 1
            
        topics_distribution = defaultdict(lambda:defaultdict(lambda:0))
        for i, e in enumerate(self.doc_dates_topics):
            topic_list = e["topic"]
            date = norm_dates[i]

            for j, v in enumerate(topic_list):
                topics_distribution[j][date] += v/dates_frequencies[date]
        
        return {k : list(sorted(v.items())) for k, v in topics_distribution.items()}