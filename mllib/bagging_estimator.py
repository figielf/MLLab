import numpy as np
from collections import Counter
from sklearn.metrics import r2_score


class BaggingBase():
  def __init__(self, model_factory_fun, n_models, sample_size=None):
    self.model_factory_fun = model_factory_fun
    self.n_models = n_models
    self.sample_size = sample_size

  def fit(self, X, Y):
    N = len(Y)
    b_sample_size = self.sample_size if self.sample_size is not None else N
    self.models = []
    for k in range(self.n_models):
      print(f'Fitting {k+1}-th model out of {self.n_models}')
      sample_idx = np.random.choice(N, size=b_sample_size, replace=True)
      Xb = X[sample_idx]
      Yb = Y[sample_idx]

      model = self.model_factory_fun()
      model.fit(Xb, Yb)
      self.models.append(model)

  def _predictions(self, X):
    predictions = []
    for model in self.models:
      pred = model.predict(X)
      predictions.append(pred)
    return np.array(predictions)


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


class BaggingRegressor(BaggingBase):
  def __init__(self, model_factory_fun, n_models, sample_size=None):
    super().__init__(model_factory_fun, n_models, sample_size)

  def predict(self, X):
    predictions = super()._predictions(X)
    return predictions.mean(axis=0)

  def score(self, X, Y):
    Y_hat = self.predict(X)
    return r2_score(Y_hat, Y)
