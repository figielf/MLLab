import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from nlp.glove_model import glove
from tests.utils.nlp_data_utils import get_sequences_with_word2idx_from_wiki_corpus, \
    get_sequences_with_word2idx_from_brown_corpus, START_STR, END_STR


def find_analogies(w1, w2, w3, We, word2idx, idx2word):
    V, D = We.shape

    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    for dist in ('euclidean', 'cosine'):
        distances = pairwise_distances(v0.reshape(1, D), We, metric=dist).reshape(V)
        # idx = distances.argmin()
        # best_word = idx2word[idx]
        idx = distances.argsort()[:4]
        best_idx = -1
        keep_out = [word2idx[w] for w in (w1, w2, w3)]
        for i in idx:
            if i not in keep_out:
                best_idx = i
                break
        best_word = idx2word[best_idx]

        print("closest match by", dist, "distance:", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)


def test_model(We, word2idx, idx2word):
    find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
    find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
    find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
    find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
    find_analogies('france', 'french', 'english', We, word2idx, idx2word)
    find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
    find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
    find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
    find_analogies('december', 'november', 'june', We, word2idx, idx2word)


def build_model(data='brown', n_vocab=5000, plot_cost=True):
    if data == 'wiki':
        sequences, word2idx, _ = get_sequences_with_word2idx_from_wiki_corpus(n_vocab=n_vocab)
    elif data == 'brown':
        keep_words = set([
            'king', 'man', 'woman',
            'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
            'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
            'australia', 'australian', 'december', 'november', 'june',
            'january', 'february', 'march', 'april', 'may', 'july', 'august',
            'september', 'october',
        ])
        sequences, word2idx, _ = get_sequences_with_word2idx_from_brown_corpus(n_vocab=n_vocab,
                                                                               include_start_end_in_vocab=True,
                                                                               keep_words=keep_words)

    V = len(word2idx)
    model = glove(100, V, 10, start_end_tokens=(word2idx[START_STR], word2idx[END_STR]))

    costs = model.fit_by_als(sequences, cc_matrix=None, n_epochs=10)

    if plot_cost:
        plt.figure(figsize=(16, 16))
        plt.plot(costs, label='cost')
        plt.legend()
        plt.show()

    return model.W, model.U, word2idx


def visualize_words(embeddings, words, word2idx):
    idx = [word2idx[w] for w in words]

    tsne = TSNE()
    Z = tsne.fit_transform(embeddings)
    Z = Z[idx]
    plt.figure(figsize=(16, 16))
    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(len(words)):
        plt.annotate(s=words[i], xy=(Z[i, 0], Z[i, 1]))
    plt.show()


if __name__ == '__main__':
    W, U, word2idx = build_model('brown', n_vocab=5000, plot_cost=False)

    idx2word = {i: w for w, i in word2idx.items()}
    print('W.shape:', W.shape)
    print('U.T.shape:', U.T.shape)

    We = np.hstack([W, U])
    We_2 = (W + U) / 2

    print('Concatenated W and U:')
    test_model(We, word2idx, idx2word)
    print('Mean of W and U:')
    test_model(We_2, word2idx, idx2word)

    # words = ['japan', 'japanese', 'england', 'english', 'australia', 'australian', 'china', 'chinese', 'italy', 'italian', 'french', 'france', 'spain', 'spanish']
    words = [
        'king', 'man', 'woman',
        'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
        'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
        'australia', 'australian', 'december', 'november', 'june',
        'january', 'february', 'march', 'april', 'may', 'july', 'august',
        'september', 'october'
    ]
    visualize_words(We, words, word2idx)
