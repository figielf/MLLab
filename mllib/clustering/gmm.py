import numpy as np
from scipy.stats import multivariate_normal


class gmm:
    def __init__(self, K):
        self.K = K
        self.pi = None
        self.mu = None
        self.sigma = None

    def fit(self, x, max_iter=100, eps=0.001, smoothing=1e-2):
        N, D = x.shape

        self.mu = np.zeros((self.K, D))
        self.sigma = np.zeros((self.K ,D, D))
        self.pi = np.ones(self.K) / self.K  # uniform initial distribution

        R = np.zeros((N, self.K))  # cluster responsibilities

        ll_history = []
        for k in range(self.K):
            self.mu[k] = x[np.random.choice(N)]
            self.sigma[k] = np.eye(D)

        for step in range(max_iter):
            # E step
            weighted_pdfs = np.zeros((N, self.K))
            for k in range(self.K):
                weighted_pdfs[:, k] = self.pi[k] * multivariate_normal.pdf(x, self.mu[k, :], self.sigma[k, :, :])
            R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

            # M step
            for k in range(self.K):
                Nk = R[:, k].sum()
                self.pi[k] = Nk / N
                self.mu[k, :] = R[:, k].dot(x) / Nk

                x_mu = x - self.mu[k, :]
                self.sigma[k, :, :] = (np.expand_dims(R[:, k], -1) * x_mu).T.dot(x_mu) / (Nk + np.eye(D) * smoothing)

            log_ll = np.log(weighted_pdfs.sum(axis=1)).sum()
            ll_history.append(log_ll)

            if step > 1:
                if np.abs(ll_history[-1] - ll_history[-2]) < eps:
                    print(f'early stop after {step} steps')
                    break

        return R, ll_history

    def predict(self, x):
        weighted_pdfs = np.zeros((len(x), self.K))
        for k in range(self.K):
            weighted_pdfs[:, k] = self.pi[k] * multivariate_normal.pdf(x, self.mu[k, :], self.sigma[k, :, :])
        R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)
        return R, R.argmax(axis=1)
