import numpy as np
from sklearn.metrics import pairwise_distances

GLOVE_EMBEDDINGS_PATH = 'C:\\dev\\my_private\\transfer_learning\\glove\\glove.6B.50d.txt'


def d_euclidean(a, b):
    return np.linalg.norm(a - b)


def d_cosine(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_nearest_neighbor_words(word, n=5, metric='euclidean', print_results=True):
    assert word in word2vec

    word_vec = word2vec[word]
    distances = pairwise_distances(word_vec.reshape(1, D), embedding, metric=metric).reshape(V)
    closed_words = []
    for idx in distances.argsort()[1:n + 1]:
        closed_words.append(idx2word[idx])

    if print_results:
        print(f'{word} -> {closed_words}')
    return closed_words


def find_analogies(w1, w2, w3, word2vec, metric='euclidean', print_results=True):
    for w in [w1, w2, w3]:
        assert w in word2vec

    analogy_vec = word2vec[w1] - word2vec[w2] + word2vec[w3]  # king - man + woman
    analogy_word = pairwise_distances(analogy_vec.reshape(1, D), embedding, metric=metric).reshape(V)
    for idx in analogy_word.argsort()[:4]:
        candidate = idx2word[idx]
        if candidate not in [w1, w2, w3]:
            if print_results:
                print(f'{w1} - {w2} + {w3} -> {candidate}')
            return candidate


if __name__ == '__main__':
    word2vec = {}
    embedding = []
    idx2word = []
    with open(GLOVE_EMBEDDINGS_PATH, encoding='utf-8') as f:  # in windows use encoding='utf-8' parameter
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            word2vec[word] = coefs
            embedding.append(coefs)
            idx2word.append(word)
    embedding = np.array(embedding)

    V, D = embedding.shape

    print(f'Found {V} word vectors of len {D}')
    print(f'As example, GloVe features for "the" word: {word2vec["the"]}\n')

    find_analogies('king', 'man', 'woman')
    find_analogies('france', 'paris', 'london')
    find_analogies('france', 'paris', 'rome')
    find_analogies('paris', 'france', 'italy')
    find_analogies('france', 'french', 'english')
    find_analogies('japan', 'japanese', 'chinese')
    find_analogies('japan', 'japanese', 'italian')
    find_analogies('japan', 'japanese', 'australian')
    find_analogies('december', 'november', 'june')
    find_analogies('miami', 'florida', 'texas')
    find_analogies('einstein', 'scientist', 'painter')
    find_analogies('china', 'rice', 'bread')
    find_analogies('man', 'woman', 'she')
    find_analogies('man', 'woman', 'aunt')
    find_analogies('man', 'woman', 'sister')
    find_analogies('man', 'woman', 'wife')
    find_analogies('man', 'woman', 'actress')
    find_analogies('man', 'woman', 'mother')
    find_analogies('heir', 'heiress', 'princess')
    find_analogies('nephew', 'niece', 'aunt')
    find_analogies('france', 'paris', 'tokyo')
    find_analogies('france', 'paris', 'beijing')
    find_analogies('february', 'january', 'november')
    find_analogies('france', 'paris', 'rome')
    find_analogies('paris', 'france', 'italy')

    find_nearest_neighbor_words('king')
    find_nearest_neighbor_words('france')
    find_nearest_neighbor_words('japan')
    find_nearest_neighbor_words('einstein')
    find_nearest_neighbor_words('woman')
    find_nearest_neighbor_words('nephew')
    find_nearest_neighbor_words('february')
    find_nearest_neighbor_words('rome')
