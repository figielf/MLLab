import numpy as np
import string
from sklearn.metrics import confusion_matrix, f1_score

from integration_tests.utils.data_utils import get_edgar_allan_and_robert_frost_data, get_robert_frost_data
from markov_models.mm_1order import MAPClassifier, SimpleMarkovModel
from markov_models.mm_2order import SparseSecondOrderMarkovModel, MAPDiscreteSequenceGenerator

END_TOKEN = '<END>'

def build_word2idx(txt_lines, skip_word_filter=None, end_token=None):
  word2idx = {'<unknown>': 0}
  if end_token is not None:
    word2idx = {end_token: 1}
  idx = len(word2idx)
  for line in txt_lines:
    for token in line.split():
      if token in word2idx:
        continue
      elif skip_word_filter is not None and not skip_word_filter(token):
        continue
      else:
        word2idx[token] = idx
        idx += 1
  return word2idx

def tokenize(txt, word2idx, end_token=None):
  vector = []
  for token in txt.split():
    vector.append(word2idx.get(token, 0))
  if end_token is not None:
    vector.append(word2idx[end_token])
  return vector

def idx2word(token_idxs, word2idx):
  idx2token = {}
  for k, v in word2idx.items():
    idx2token[v] = k

  return [idx2token[idx] for _, idx in enumerate(token_idxs)]


if __name__ == '__main__':
    X_train = np.array(get_robert_frost_data())

    word2idx_train = build_word2idx(X_train, end_token=END_TOKEN)
    V = len(word2idx_train)
    print('vocab size V:', V)

    X_train_vectorized = np.array([tokenize(line, word2idx_train, end_token=END_TOKEN) for line in X_train],
                                  dtype=object)
    seq_model = SparseSecondOrderMarkovModel(V, seq_end=word2idx_train[END_TOKEN])
    seq_model.fit(X_train_vectorized)
    generator = MAPDiscreteSequenceGenerator(seq_model)

    print('\nArtificially created poem lines:')
    print('\t', ' '.join(idx2word(generator.generate(3), word2idx_train)))
    print('\t', ' '.join(idx2word(generator.generate(5), word2idx_train)))
    print('\t', ' '.join(idx2word(generator.generate(10), word2idx_train)))
    print('\t', ' '.join(idx2word(generator.generate(30), word2idx_train)))