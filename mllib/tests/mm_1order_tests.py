import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from tests.utils.data_utils import get_edgar_allan_and_robert_frost_data
from markov_models.mm_1order import MAPClassifier, SimpleMarkovModel


def build_word2idx(txt_lines, skip_word_filter=None):
  word2idx = {'<unknown>': 0}
  idx = 1
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

def tokenize(txt, word2idx):
  vector = []
  for token in txt.split():
    vector.append(word2idx.get(token, 0))
  return vector




if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_edgar_allan_and_robert_frost_data()

    word2idx_train = build_word2idx(X_train)
    V = len(word2idx_train)
    print('vocab size V:', V)

    X_train_vectorized = np.array([tokenize(line, word2idx_train) for line in X_train], dtype=object)
    X_test_vectorized = np.array([tokenize(line, word2idx_train) for line in X_test], dtype=object)

    mm = MAPClassifier([SimpleMarkovModel(V), SimpleMarkovModel(V)])
    mm.fit(X_train_vectorized, y_train)

    pred_train = mm.predict(X_train_vectorized)
    print(f'train accuracy: {(pred_train == y_train).mean()}')
    print(f'train F1 score: {f1_score(y_train, pred_train)}')
    print('train confusion matrix:')
    print(confusion_matrix(y_train, pred_train))

    pred_test = mm.predict(X_test_vectorized)
    print(f'test accuracy: {(pred_test == y_test).mean()}')
    print(f'test F1 score: {f1_score(y_test, pred_test)}')
    print('test confusion matrix:')
    print(confusion_matrix(y_test, pred_test))