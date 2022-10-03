import numpy as np
from scores import exponential_loss


class AdaBoostClassifier:
    def __init__(self, base_model_factory, n_steps):
        self.base_model_factory = base_model_factory
        self.n_steps = n_steps
        self.base_models = []
        self.alphas = []

    def fit(self, X, Y):
        N = len(X)
        w = np.ones(N) / N

        for step in range(self.n_steps):
            f = self.base_model_factory()
            f.fit(X, Y, sample_weight=w)
            y_hat = f.predict(X)

            eps = w.dot(y_hat != Y)  # / w.sum() # this is already normalized below
            alpha = 0.5 * (np.log(1 - eps) - np.log(eps))
            w = w * np.exp(-alpha * Y * y_hat)
            w = w / np.sum(w)  # normalize w

            self.alphas.append(alpha)
            self.base_models.append(f)

    def predict(self, X):
        return np.sign(self.predict_p(X))

    def predict_p(self, X):
        y_hat = np.zeros(len(X))
        for f, alpha in zip(self.base_models, self.alphas):
            y_hat += alpha * f.predict(X)
        return y_hat

    def score(self, X, Y):
        f_hat = self.predict(X)
        return (f_hat == Y).mean()

    def score2(self, X, Y):
        f_hat = self.predict_p(X)
        score = (np.sign(f_hat) == Y).mean()
        loss = exponential_loss(Y, f_hat)
        return score, loss
