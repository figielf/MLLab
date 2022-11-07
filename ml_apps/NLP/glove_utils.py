import os

import numpy as np

from data_utils import get_data_dir


def load_glove_embeddings(glove_folder, embedding_dim):
    # load in pre-trained word vectors

    path = os.path.join(glove_folder, f'glove.6B.{embedding_dim}d.txt')
    print(f'Loading glove word vectors from path {path} ...')
    word2vec = {}
    with open(path, encoding='utf-8') as f:
        # is just a space-separated text file in the format word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print(f'Loaded {len(word2vec)} GLOVE word vectors.')
    return word2vec


def get_embedding_matrix(vocab_size, word2vec, word2idx):
    vec_iter = iter(word2vec.values())
    embedding_dim = len(next(vec_iter))
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        if idx < vocab_size:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[idx] = embedding_vector
    print(f'GLOVE initialization of embeddings matrix shape: {embedding_matrix.shape}')
    return embedding_matrix


def get_robert_frost_data_for_seq2seq(start_token='<sos>', end_token='<eos>'):
    input_texts = []
    target_texts = []
    with open(get_data_dir('robert_frost.txt'), encoding='utf-8') as file:
        for line in file:
            txt = line.rstrip().lower()
            if txt:
                #txt = txt.replace('\\n', '')
                input_texts.append(start_token + ' ' + txt)
                target_texts.append(txt + ' ' + end_token)
    return input_texts, target_texts