from gensim.models import KeyedVectors

WORD2VEC_EMBEDDINGS_PATH = 'C:\\dev\\my_private\\transfer_learning\\word2vec\\GoogleNews-vectors-negative300.bin'


def find_nearest_neighbor_words(word2vec_word_vectors, word, n=5, print_results=True):
    closed_words = [w for w, score in word2vec_word_vectors.most_similar(positive=[word], topn=n)]
    if print_results:
        print(f'{word} -> {closed_words}')
    return closed_words


def find_analogies(word2vec_word_vectors, w1, w2, w3, print_results=True):
    closed_words = word2vec_word_vectors.most_similar(positive=[w1, w3], negative=[w2])
    if print_results:
        print(f'{w1} - {w2} + {w3} -> {closed_words[0][0]}')
    return closed_words[0][0],


if __name__ == '__main__':
    word_vectors = KeyedVectors.load_word2vec_format(WORD2VEC_EMBEDDINGS_PATH, binary=True)

    D = word_vectors.vector_size
    V = len(word_vectors)
    print(f'Found {V} word vectors of len {D}')
    print(f'As example, GloVe features for "the" word: {word_vectors["the"]}\n')

    find_analogies(word_vectors, 'king', 'man', 'woman')
    find_analogies(word_vectors, 'france', 'paris', 'london')
    find_analogies(word_vectors, 'france', 'paris', 'rome')
    find_analogies(word_vectors, 'paris', 'france', 'italy')
    find_analogies(word_vectors, 'france', 'french', 'english')
    find_analogies(word_vectors, 'japan', 'japanese', 'chinese')
    find_analogies(word_vectors, 'japan', 'japanese', 'italian')
    find_analogies(word_vectors, 'japan', 'japanese', 'australian')
    find_analogies(word_vectors, 'december', 'november', 'june')
    find_analogies(word_vectors, 'miami', 'florida', 'texas')
    find_analogies(word_vectors, 'einstein', 'scientist', 'painter')
    find_analogies(word_vectors, 'china', 'rice', 'bread')
    find_analogies(word_vectors, 'man', 'woman', 'she')
    find_analogies(word_vectors, 'man', 'woman', 'aunt')
    find_analogies(word_vectors, 'man', 'woman', 'sister')
    find_analogies(word_vectors, 'man', 'woman', 'wife')
    find_analogies(word_vectors, 'man', 'woman', 'actress')
    find_analogies(word_vectors, 'man', 'woman', 'mother')
    find_analogies(word_vectors, 'heir', 'heiress', 'princess')
    find_analogies(word_vectors, 'nephew', 'niece', 'aunt')
    find_analogies(word_vectors, 'france', 'paris', 'tokyo')
    find_analogies(word_vectors, 'france', 'paris', 'beijing')
    find_analogies(word_vectors, 'february', 'january', 'november')
    find_analogies(word_vectors, 'france', 'paris', 'rome')
    find_analogies(word_vectors, 'paris', 'france', 'italy')

    find_nearest_neighbor_words(word_vectors, 'king')
    find_nearest_neighbor_words(word_vectors, 'france')
    find_nearest_neighbor_words(word_vectors, 'japan')
    find_nearest_neighbor_words(word_vectors, 'einstein')
    find_nearest_neighbor_words(word_vectors, 'woman')
    find_nearest_neighbor_words(word_vectors, 'nephew')
    find_nearest_neighbor_words(word_vectors, 'february')
    find_nearest_neighbor_words(word_vectors, 'rome')


