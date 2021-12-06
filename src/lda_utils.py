import numpy as np

def normalize_dict(d, precision=3):
    
    tot = sum(d.values())
    if tot == 0: return d
    return {k : round(v*100/tot, precision) for k,v in d.items()}
    
def get_word_relevance(word, word2id, vocab, lda_model, normalize=False, precision=3):
    
    if word not in word2id:
        return {i : 0 for i in range(len(lda_model.components_))}
    else:
        ret = {i : comp[word2id[word]] 
               for i, comp in enumerate(lda_model.components_)}
        return normalize_dict(ret) if normalize else ret
    
def get_words_relevance(words, word2id, vocab, lda_model, normalize=False, precision=3):
    
    ret = {i : 0 for i in range(len(lda_model.components_))}
    for word in words:
        for t, value in get_word_relevance(word, word2id, vocab, lda_model).items():
            ret[t] += value
            
    return normalize_dict(ret) if normalize else ret

def print_topics(model, vectorizer, n_top_words=10, only_interesting=False, interesting_set={}):
    vocab = vectorizer.get_feature_names()
    topic_words = {}
    for topic, comp in enumerate(model.components_): 
        if only_interesting:
            word_idx = np.argsort(comp)[::-1]
            topic_words[topic] = [el for el in [(vocab[i], comp[i]) for i in word_idx] 
                                  if el[0] in interesting_set][:n_top_words]
        else: 
            word_idx = np.argsort(comp)[::-1][:n_top_words]
            topic_words[topic] = [(vocab[i], comp[i]) for i in word_idx]        

    for topic, words in topic_words.items():
        print('\nTopic: %d' % topic)
        s = ""
        for w, r in words:
            s += f"{round(r, 2)}*{w} + "
        print(s[:-3])