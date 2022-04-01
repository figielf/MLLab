import numpy as np
from matplotlib import pyplot as plt

from clustering.clustering_evaluation import calc_purity_soft_hist, calc_davis_bouldin_index_soft_hist
from clustering.kmeans_soft import Kmeans_soft
from integration_tests.utils.data_utils import get_mnist_data
from integration_tests.utils.plot_utils import plot_clusters, plot_clusters_by_weights, plot_clusters_by_weights_history


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


def report_on_kmeans(model, x, y, clusters_hist, r_hist, centres_hist, cost_hist, include_plot=True):
    if include_plot:
        plot_clusters_by_weights(X, r_hist[-1], centres_hist, color_base=BASE_COLOR)

    plt.figure(figsize=(10, 5))
    plt.plot(cost_hist)
    plt.title('cost function history')
    plt.show()

    purity_hist = calc_purity_soft_hist(y, r_hist)
    plt.figure(figsize=(10, 5))
    plt.plot(purity_hist)
    plt.title('purity cost function history')
    plt.show()

    purity_hist = calc_davis_bouldin_index_soft_hist(x, r_hist, centres_hist[1:], k_means._norm_func)
    plt.figure(figsize=(10, 5))
    plt.plot(purity_hist)
    plt.title('davis-bouldin index cost function history')
    plt.show()

    purity_score = model.score(x, y, method='purity')
    dbi_score = model.score(x, y, method='dbi')
    print(f'final purity: {purity_score}')
    print(f'final davis-bouldin index: {dbi_score}')

    if include_plot:
        plot_clusters_by_weights_history(x, clusters_hist, r_hist, centres_hist[:-1], color_base=BASE_COLOR)


if __name__ == '__main__':
    # k means on generatd data
    K = 3
    BASE_COLOR = np.random.random((K, 3))
    X, y, centres0 = get_cloud_data(K, N=900, D=2)

    k_means = Kmeans_soft(n_clusters=K)
    clusters_hist, r_hist, centres_hist, cost_hist = k_means.fit(X, beta=1.0, initial_centres=centres0, max_steps=100, logging_step=10)
    report_on_kmeans(k_means, X, y, clusters_hist, r_hist, centres_hist, cost_hist)

    # k means on MNIST data
    Kmnist = 10
    Nmnist = 1000

    Xmnist, _, Ymnist, _, picture_shape = get_mnist_data(train_size=0.8, should_plot_examples=True)
    Xmnist = Xmnist[:Nmnist]
    Ymnist = Ymnist[:Nmnist]

    shuffle_ids = np.arange(Nmnist)
    np.random.shuffle(shuffle_ids)
    centres0_mnist = Xmnist[shuffle_ids[:Kmnist]]

    k_means = Kmeans_soft(n_clusters=Kmnist)
    mnist_clusters_hist, mnist_r_hist, mnist_centres_hist, mnist_cost_hist = k_means.fit(Xmnist, centres0_mnist,
                                                                                          beta=1.0, max_steps=100,
                                                                                          logging_step=10)
    report_on_kmeans(k_means, Xmnist, Ymnist, mnist_clusters_hist, mnist_r_hist, mnist_centres_hist, mnist_cost_hist, include_plot=False)

    # plot the mean images
    # they should look like digits
    for k in range(len(mnist_centres_hist[-1])):
        im = mnist_centres_hist[-1][k].reshape(28, 28)
        plt.imshow(im, cmap='gray')
        plt.show()



