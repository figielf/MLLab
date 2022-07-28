import json
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from nlp.word2vec_model import word2vec
from nlp.word2vec_model_tf import word2vec_tf
from tests.utils.nlp_data_utils import get_sequences_with_word2idx_from_brown_corpus, get_idx2word_mapping, \
    get_sequences_with_word2idx_from_wiki_corpus


def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, embeddings):
    V, D = embeddings.shape

    print(f'testing: {pos1} - {neg1} = {pos2} - {neg2}')
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print(f'Sorry, {w} not in word2idx')
            return

    p1 = embeddings[word2idx[pos1]]
    n1 = embeddings[word2idx[neg1]]
    p2 = embeddings[word2idx[pos2]]
    n2 = embeddings[word2idx[neg2]]
    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), embeddings, metric="cosine").reshape(V)
    idx = distances.argsort()[:10]

    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    for i in idx:
        if i not in keep_out:
            best_idx = i
            break

    print(f'got: {pos1} - {neg1} = {idx2word[best_idx]} - {neg2}')
    print('closest 10:')
    for i in idx:
        print(idx2word[i], distances[i])
    print(f'dist to {pos2}: {cosine_similarity(p2.reshape(1, -1), vec.reshape(1, -1))}')
    print(f'dist to {pos2}: {pairwise_distances(p2.reshape(1, -1), vec.reshape(1, -1), metric="cosine")}')


def save_model(folder_name, W1, W2, word2idx):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    with open(f'{folder_name}/w2v_word2idx.json', 'w') as f:
        json.dump(word2idx, f)

    np.savez(f'{folder_name}/w2v_model.npz', W1, W2)


def load_model(folder_name):
    with open(f'{folder_name}/w2v_word2idx.json') as f:
        word2idx = json.load(f)
    npz = np.load(f'{folder_name}/w2v_model.npz')
    W1 = npz['arr_0']
    W2 = npz['arr_1']
    return W1, W2, word2idx


def test_model(word2idx, W, V):
    idx2word = get_idx2word_mapping(word2idx)

    # there are multiple ways to get the "final" word embedding
    # We = W
    # We = V.T
    # We = (W + V.T) / 2
    for We in (W,): # V.T, (W + V.T) / 2):
        print("**********")
        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
        analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
        analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
        analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
        analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
        analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
        analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
        analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)


def build_model(data='wiki', n_vocab=20000, with_tensorflow=False):
    if data == 'wiki':
        sequences, word2idx, _ = get_sequences_with_word2idx_from_wiki_corpus(n_vocab=n_vocab)
    elif data == 'brown':
        sequences, word2idx, _ = get_sequences_with_word2idx_from_brown_corpus(n_vocab=n_vocab, include_start_end=False,
                                                                               keep_words=None)
    else:
        raise Exception('Only wiki or brown corpus is supported')

    save_path = 'c:/temp/' + data

    V = len(word2idx)
    print("Vocab size:", V)

    # config
    epochs = 20
    context_size = 2  # half size -> window is of size 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    learning_rate_decay = (learning_rate - final_learning_rate) / epochs
    D = 50  # word embedding size

    if with_tensorflow:
        word2vec_model = word2vec_tf(D, word2idx, context_size=context_size)
        W1, W2, word2vec_model_loss = word2vec_model.fit(sequences, word2idx, learning_rate=learning_rate,
                                                     learning_rate_decay=learning_rate_decay, n_epochs=epochs)
    else:
        word2vec_model = word2vec(D, word2idx, context_size=context_size)
        W1, W2, word2vec_model_loss = word2vec_model.fit(sequences, word2idx, learning_rate=learning_rate,
                                                     learning_rate_decay=learning_rate_decay, n_epochs=epochs)
    save_model(save_path, W1, W2, word2idx)

    plt.figure(figsize=(20, 16))
    plt.plot(word2vec_model_loss, label='word2vec model fit loss')
    plt.legend()
    plt.show()
    return W1, W2, word2idx


if __name__ == '__main__':
    W1, W2, word2idx = build_model('brown', n_vocab=2000, with_tensorflow=False)
    #W1, W2, word2idx = load_model('C:/dev/my_private/machine_learning_examples/nlp_class2')
    test_model(word2idx, W1, W2)
