import numpy as np


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
