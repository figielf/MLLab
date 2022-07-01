import numpy as np

from clustering.clustering_evaluation import purity_cost, davis_bouldin_index_cost
from tests.utils.data_utils import RANDOM_STATE

norm_vec = np.vectorize(lambda x: x.dot(x), signature='(d)->()')


class Kmeans_hard:
    def __init__(self, n_clusters, norm_func=norm_vec):
        self._n_clusters = n_clusters
        self._norm_func = norm_func
        self._X = None
        self.cluster_centres = None

    def fit(self, X, initial_centres=None, max_steps=None, logging_step=None):
        assert X.shape[1] == initial_centres.shape[1]
        if initial_centres is None:
            shuffle_ids = np.arange(len(X))
            np.random.shuffle(shuffle_ids)
            initial_centres = X[shuffle_ids[:self._n_clusters]]

        centres_hist = [initial_centres]
        clusters_hist = []
        cost_hist = []
        step = 0
        while True:
            if logging_step is not None and step % logging_step == 0:
                print('step:', step)
            if max_steps is not None and step > max_steps:
                print(f'early stop after {step + 1} steps - reached maximum number of steps')
                break
            new_clusters, new_centres, cost = self._k_means_step(X, centres_hist[-1])
            clusters_hist.append(new_clusters)
            centres_hist.append(new_centres)
            cost_hist.append(cost)

            if step > 1 and np.array_equal(prev_clusters, new_clusters):
                print(f'early stop after {step + 1} steps - assigned clusters has not changed')
                break

            prev_clusters: None = new_clusters
            step += 1

        self.cluster_centres = centres_hist[-1]
        return clusters_hist, centres_hist, cost_hist

    def predict(self, X):
        assert self.cluster_centres is not None
        return self._assign_clusters(X, self.cluster_centres)

    def score(self, X, Y, method='purity'):
        clusters, cost = self.predict(X)
        if method == 'purity':
            return purity_cost(Y, clusters)
        elif method == 'dbi':
            return davis_bouldin_index_cost(X, clusters, self.cluster_centres, self._norm_func)
        else:
            raise Exception('Incorrect score type, should be "purity" or "dbi"')

    @staticmethod
    def _find_centres(x, y):
        assert x.shape[0] == y.shape[0]

        n_classes = len(set(y))
        mu = np.zeros((n_classes, x.shape[1]))
        for k in range(n_classes):
            mu[k, :] = x[y == k].mean(axis=0)
        return mu

    norm_vec = np.vectorize(lambda x: x.dot(x), signature='(d)->()')

    @staticmethod
    def _assign_clusters(x, centres):
        assert x.shape[1:] == centres.shape[1:]
        N = x.shape[0]
        K = centres.shape[0]

        d = np.zeros((N, K))
        for c in range(centres.shape[0]):
            diff = x - centres[c]
            d[:, c] = norm_vec(diff)

        closest_centre = d.argmin(axis=1)
        cost = d[np.arange(N), closest_centre].sum()
        return closest_centre, cost

    def _k_means_step(self, x, centres):
        new_clusters, cost = self._assign_clusters(x, centres)
        new_centres = self._find_centres(x, new_clusters)
        return new_clusters, new_centres, cost
