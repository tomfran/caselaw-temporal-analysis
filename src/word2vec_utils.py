# credits: https://gist.github.com/zhicongchen/9e23d5c3f1e5b1293b16133485cd17d8
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import os

def print_similar(word, models, n=5):
    print(word)
    for y, m in models.items():
        try:
            print(f"\t{y}: {[e[0] for e in m.wv.most_similar(word, topn=n)]}")
        except:
            pass
        
def cos_sim(a, b):
    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)

def get_similarity(m1, m2, word):
    try:
        return cos_sim(m1.wv[word], m2.wv[word])
    except:
        return -1
    
def get_similarity_sequence_base(models, base, word):
    return [get_similarity(models[base], models[e], word) 
            for e in sorted(models.keys(), reverse=True)]

def get_similarity_sequence_consecutive(models, word):
    k = list(sorted(models.keys(), reverse=True))
    couples = zip(k, k[1:])
    return [get_similarity(models[s], models[e], word) for s, e in couples] 

def order_by_semantic_shift(words, models, base="2010", interval=-1):
    res = []
    for word in words:
        s = get_similarity_sequence(models, base, word)[:interval]
        s = [e for e in s if e>0]
        if s:
            index = np.argmin(s)
            res.append((word, years[index], s[index]))
        else:
            print(word, "not in models")
        
    res.sort(key=lambda x : x[2])
    return res

def align_models(models):
    k = list(sorted(models.keys(), reverse=True))
    couples = zip(k, k[1:])
    for base, other in couples:
        smart_procrustes_align_gensim(models[base], models[other])

def load_models(path):

    def get_name(s):
        s = s.split("/")[-1]
        return int(s.split("_")[0])

    return { get_name(model_name) : Word2Vec.load(model_name) 
            for model_name in [f"{path}/{el}" 
                               for el in sorted(os.listdir(path)) if "npy" not in el]}

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # re-filling the normed vectors
    in_base_embed.wv.fill_norms(force=True)
    in_other_embed.wv.fill_norms(force=True)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        # print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)

def show_sequence_plot_base(word, base, models):
    fig, axs = plt.subplots(1,1, figsize=(15,5))

    x = [e for e in get_similarity_sequence_base(models, base, word) if e>0]

    axs.plot(x)
    plt.title(f"Word: {word}, base reference: {base}")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    
    years = list(sorted(models.keys(), reverse=True))
    plt.xticks(ticks=range(len(x)), labels=years[:len(x)], rotation=90)
    plt.yticks(ticks=[i/10 for i in range(11)])

    plt.axhline(y=0.7, color='r', linestyle='-')
    plt.show()
    
def show_sequence_plot_epochs(word, models):
    fig, axs = plt.subplots(1,1, figsize=(15,5))

    x = [e for e in get_similarity_sequence_consecutive(models, word) if e>0]

    axs.plot(x)
    plt.title(f"Word: {word}")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    
    years = list(sorted(models.keys(), reverse=True))
    plt.xticks(ticks=range(len(x)), labels=years[:len(x)], rotation=90)
    plt.yticks(ticks=[i/10 for i in range(11)])

    plt.axhline(y=0.7, color='r', linestyle='-')
    plt.show()
    