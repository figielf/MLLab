from sklearn.metrics import r2_score
from bagging_base import BaggingBase


class BaggingRegressor(BaggingBase):
  def __init__(self, model_factory_fun, n_models, sample_size=None):
    super().__init__(model_factory_fun, n_models, sample_size)

  def predict(self, X):
    predictions = super()._predictions(X)
    return predictions.mean(axis=0)

  def score(self, X, Y):
    Y_hat = self.predict(X)
    return r2_score(Y_hat, Y)
