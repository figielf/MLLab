import numpy as np


class pca:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.explained_variance_ratio = np.array([0])
        self.singular_vector_matrix = None

    def fit(self, X):
        cov = np.cov(X.T)
        lambdas, Q = np.linalg.eigh(cov)
        sort_idx = np.argsort(-lambdas)
        lambdas = np.maximum(lambdas[sort_idx], 0)  # values can be slightly negative
        self.explained_variance_ratio = lambdas / np.sum(lambdas)
        self.singular_vector_matrix = Q[:, sort_idx]

        if self.n_components is not None:
            self.explained_variance_ratio = self.explained_variance_ratio[:self.n_components]
            self.singular_vector_matrix = self.singular_vector_matrix[:, :self.n_components]

    def transform(self, X):
        return X.dot(self.singular_vector_matrix)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
