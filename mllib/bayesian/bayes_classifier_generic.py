import numpy as np


class bayes_classifier:
    def __init__(self, n_classes, model_factory):
        self.K = n_classes
        self.model_factory = model_factory
        self.priors = None
        self.models = None
        self.D = None

    def fit(self, X, Y):
        self.D = X.shape[1]
        self.priors = []
        self.models = []
        for k in range(self.K):
            model = self.model_factory()
            print(f'Fitting {k + 1}/{self.K} {model} model...')
            class_filter = Y == k
            self.priors.append(np.mean(class_filter))
            model.fit(X[class_filter], Y[class_filter])
            self.models.append(model)

    def predict(self, X):
        assert X.shape[1] == self.D

        ll = np.zeros((len(X), self.K))
        for k in range(self.K):
            ll[:, k] = self.models[k].predict_proba(X)
        best_k = np.argmax(ll, axis=1)
        bext_ll = ll[:, best_k]
        return best_k, bext_ll

    def score(self, X, Y):
        y_hat, _ = self.predict(X)
        return np.mean(y_hat == Y)

    def sample_given_y(self, k):
        model = self.models[k]
        return model.sample()

    def sample(self):
        prior_class = np.random.choice(self.K, p=self.priors)
        return self.sample_given_y(prior_class), prior_class
