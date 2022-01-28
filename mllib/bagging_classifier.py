import numpy as np
from collections import Counter
from bagging_base import BaggingBase


class BaggingClassifier(BaggingBase):
  def __init__(self, model_factory_fun, n_models, sample_size=None):
    super().__init__(model_factory_fun, n_models, sample_size)

  def predict(self, X):
    predictions = super()._predictions(X)
    if (len(predictions[0].shape) > 1):
      raise Exception(
        f'Only non sparse prediction output is supported. Shape of single model prediction is:{predictions[0].shape}')
    else:
      N = len(X)
      Y_hat = np.zeros(N)
      for i in range(N):
        Y_hat[i] = self._most_frequent(predictions[:, i])
      return Y_hat

  def score(self, X, Y):
    Y_hat = self.predict(X)
    return (Y_hat == Y).mean()

  def _most_frequent(self, array1d):
    occurence_count = Counter(array1d)
    return occurence_count.most_common(1)[0][0]
