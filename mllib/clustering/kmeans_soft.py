import numpy as np

from clustering.clustering_evaluation import purity_soft_cost, davis_bouldin_index_soft_cost

norm_vec = np.vectorize(lambda x: x.dot(x), signature='(d)->()')


class Kmeans_soft:
    def __init__(self, n_clusters, norm_func=norm_vec):
        self._n_clusters = n_clusters
        self._norm_func = norm_func
        self._X = None
        self._beta = None
        self.cluster_responsibilities = None
        self.cluster_centres = None

    def fit(self, x, initial_centres, beta, max_steps=None, logging_step=None):
        assert x.shape[1] == initial_centres.shape[1]
        self._beta = beta

        centres_hist = [initial_centres]
        weights_hist = []
        clusters_hist = []
        cost_hist = []
        step = 0
        while True:
            if logging_step is not None and step % logging_step == 0:
                print('step:', step)
            if max_steps is not None and step > max_steps:
                print(f'early stop after {step + 1} steps - reached maximum number of steps')
                break
            clusters, new_centres, new_weights, cost = self._k_means_soft_step(x, centres_hist[-1])
            clusters_hist.append(clusters)
            centres_hist.append(new_centres)
            weights_hist.append(new_weights)
            cost_hist.append(cost)
            step += 1

            if step > 1 and np.array_equal(clusters_hist[-2], clusters_hist[-1]):
                print(f'early stop after {step} steps - assigned clusters has not changed')
                break

        self.cluster_responsibilities = weights_hist[-1]
        self.cluster_centres = centres_hist[-1]
        return clusters_hist, weights_hist, centres_hist, cost_hist

    def predict(self, X):
        assert self.cluster_centres is not None
        assert self.cluster_responsibilities is not None
        return self._assign_cluster_responsibilities(X, self.cluster_centres)

    def score(self, X, Y, method='purity'):
        clusters, r, cost = self.predict(X)
        if method == 'purity':
            return purity_soft_cost(Y, r)
        elif method == 'dbi':
            return davis_bouldin_index_soft_cost(X, r, self.cluster_centres, self._norm_func)
        else:
            raise Exception('Incorrect score type, should be "purity" or "dbi"')

    @staticmethod
    def _find_centres(x, r):
        assert x.shape[0] == r.shape[0]
        mu = r.T.dot(x) / r.sum(axis=0, keepdims=True).T
        return mu

    norm_vec = np.vectorize(lambda x: x.dot(x), signature='(d)->()')


    def _assign_cluster_responsibilities(self, x, centres):
        assert x.shape[1:] == centres.shape[1:]
        N = x.shape[0]
        K = centres.shape[0]

        d = np.zeros((N, K))
        exp_ = np.zeros((N, K))
        for k in range(centres.shape[0]):
            diff = x - centres[k]
            d[:, k] = norm_vec(diff)
            exp_[:, k] = np.exp(-self._beta * d[:, k])

        r = exp_ / exp_.sum(axis=1, keepdims=True)
        closest_centre = r.argmax(axis=1)
        cost = np.sum(r * d)

        return closest_centre, r, cost

    def _k_means_soft_step(self, x, centres):
        clusters, weights, cost = self._assign_cluster_responsibilities(x, centres)
        new_centres = self._find_centres(x, weights)
        return clusters, new_centres, weights, cost
