import numpy as np

from dimensionality_reduction.pca import pca
from gaussian_nb_classifier import gaussian_nb_classifier


class gaussian_nb_with_pca_classifier:
    def __init__(self, n_classes, n_components=None):
        self.nb_classifier = gaussian_nb_classifier(n_classes=n_classes)
        self.pca_model = pca(n_components=n_components)

    def fit(self, X, Y):
        X_decorrelated = self.pca_model.fit_transform(X)
        self.nb_classifier.fit(X_decorrelated, Y)

    def predict(self, X):
        X_decorrelated = self.pca_model.transform(X)
        y_hat = self.nb_classifier.predict(X_decorrelated)
        return y_hat

    def score(self, X, Y):
        y_hat, _ = self.predict(X)
        return np.mean(y_hat == Y)
