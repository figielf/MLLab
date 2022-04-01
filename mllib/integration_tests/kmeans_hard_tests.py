import numpy as np
from matplotlib import pyplot as plt

from clustering.clustering_evaluation import calc_purity_hist, calc_davis_bouldin_index_hist
from clustering.kmeans_hard import Kmeans_hard
from integration_tests.utils.plot_utils import plot_clusters, plot_clusters_history


def get_cloud_data(K, N, D):
    shift = 4

    mu0 = np.array([0, 0])
    mu1 = np.array([0, shift])
    mu2 = np.array([shift, shift])

    X = np.random.randn(N, D)
    X[:300] += mu0
    X[300:600] += mu1
    X[600:] += mu2

    y = np.array([0] * 300 + [1] * 300 + [2] * 300)

    ids = np.arange(N)
    np.random.shuffle(ids)
    X = X[ids]
    y = y[ids]

    plot_clusters(X, y)

    # plot centres
    shuffle_ids = np.arange(N)
    np.random.shuffle(shuffle_ids)
    centres = X[shuffle_ids[:K]]

    return X, y, centres


def report_on_kmeans(model, x, y, clusters_hist, centres_hist, cost_hist):
    plot_clusters(X, clusters_hist[-1], centres_hist)

    plt.figure(figsize=(10,5))
    plt.plot(cost_hist)
    plt.title('cost function history')
    plt.show()

    purity_hist = calc_purity_hist(y, clusters_hist)
    plt.figure(figsize=(10,5))
    plt.plot(purity_hist)
    plt.title('purity cost function history')
    plt.show()

    purity_hist = calc_davis_bouldin_index_hist(x, clusters_hist, centres_hist[1:], k_means._norm_func)
    plt.figure(figsize=(10,5))
    plt.plot(purity_hist)
    plt.title('davis-bouldin index cost function history')
    plt.show()

    purity_score = model.score(x, y, method='purity')
    dbi_score = model.score(x, y, method='dbi')
    print(f'final purity: {purity_score}')
    print(f'final davis-bouldin index: {dbi_score}')

    plot_clusters_history(x, clusters_hist, centres_hist[:-1])


if __name__ == '__main__':
    K = 3
    BASE_COLOR = np.random.random((K, 3))
    X, y, centres0 = get_cloud_data(K, N=900, D=2)

    k_means = Kmeans_hard(n_clusters=K)
    clusters_hist, centres_hist, cost_hist = k_means.fit(X, centres0, max_steps=100, logging_step=10)
    report_on_kmeans(k_means, X, y, clusters_hist, centres_hist, cost_hist)
