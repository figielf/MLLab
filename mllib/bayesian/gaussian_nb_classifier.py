import numpy as np
from scipy.stats import multivariate_normal as mvn


class gaussian_nb_classifier:
    def __init__(self, n_classes):
        self.K = n_classes
        self.D = None
        self.prior = None
        self.Mu = None
        self.Sigma2 = None

    def fit(self, X, Y, smoothing=1e-2):
        self.D = X.shape[1]
        self.prior = []
        self.Mu = []
        self.Sigma2 = []
        for k in range(self.K):
            class_filter = Y == k
            self.prior.append(np.mean(class_filter))
            self.Mu.append(np.mean(X[class_filter], axis=0))
            self.Sigma2.append(np.var(X[class_filter], axis=0) + smoothing)

    def predict(self, X):
        assert X.shape[1] == self.D

        ll = np.zeros((len(X), self.K))
        for k in range(self.K):
            ll[:, k] = mvn.logpdf(X, mean=self.Mu[k], cov=self.Sigma2[k]) + np.log(self.prior[k])
        best_k = np.argmax(ll, axis=1)
        bext_ll = ll[:, best_k]
        return best_k, bext_ll

    def score(self, X, Y):
        y_hat, _ = self.predict(X)
        return np.mean(y_hat == Y)
