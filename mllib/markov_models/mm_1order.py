import numpy as np


class IDiscreteProbabilityModel:
    def fit(self, x):
        pass

    def get_log_joint_prob(self, x):
        pass


class SimpleMarkovModel(IDiscreteProbabilityModel):
    def __init__(self, n_states):
        self.n_states = n_states
        self.log_pi = None
        self.log_A = None

    def fit(self, x, smoothing=10e-2):
        if smoothing is None:
            smoothing = 1
        pi_count = np.ones(self.n_states) * smoothing  # t_0 probability distirubtion of initial state of s
        A_count = np.ones((self.n_states, self.n_states)) * smoothing  # A[i, j] - prob of transition from state s[i] to state s[j]
        for idx, x_cur in enumerate(x):
            pi_count[x_cur[0]] += 1
            for t in range(1, len(x_cur)):
                A_count[x_cur[t - 1], x_cur[t]] += 1

        pi = pi_count / pi_count.sum()
        A = A_count / A_count.sum(axis=1, keepdims=True)

        self.log_pi = np.log(pi)
        self.log_A = np.log(A)

    def get_log_joint_prob(self, x):
        log_ll = self.log_pi[x[0]]
        for t in range(1, len(x)):
            log_ll += self.log_A[x[t - 1], x[t]]
        return log_ll


class MAPClassifier:  # Maximum Posteriori
    def __init__(self, likelihood_models, smoothing=1e-08):
        self.n_classes = len(likelihood_models)
        self.models = likelihood_models
        self.log_prior = None

    def fit(self, x, y):
        for k in range(self.n_classes):
            self.models[k].fit(x[y == k])
        self.fit_prior(y)

    def fit_prior(self, y):
        prior = np.zeros(self.n_classes)
        for k in range(self.n_classes):
            prior[k] = np.mean(y == k)
        self.log_prior = np.log(prior)

    def predict(self, x):
        n = len(x)
        posterior = np.zeros((n, self.n_classes))
        for i, x_cur in enumerate(x):
            for k in range(self.n_classes):
                posterior[i, k] = self.models[k].get_log_joint_prob(x_cur) + self.log_prior[k]
        return posterior.argmax(axis=1)
