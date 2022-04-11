import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from clustering.clustering_evaluation import purity_soft_cost, davis_bouldin_index_soft_cost
from clustering.gmm import gmm
from integration_tests.utils.data_utils import get_mnist_data
from integration_tests.utils.plot_utils import plot_clusters


def get_cloud_data(K, D):
    shift = 4

    n0 = 1200
    n1 = 1800
    n2 = 200
    N = n0 + n1 + n2

    mu0 = np.array([0, 0])
    mu1 = np.array([shift, shift])
    mu2 = np.array([0, shift])

    sigma0 = 2
    sigma1 = 1
    sigma2 = 0.5

    X = np.random.randn(N, D)
    X[:n0, :] = X[:n0, :] * sigma0 + mu0
    X[n0:(n0+n1), :] = X[n0:(n0+n1), :] * sigma1 + mu1
    X[(n0+n1):, :] = X[(n0+n1):, :] * sigma2 + mu2

    y = np.array([0] * n0 + [1] * n1 + [2] * n2)

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


if __name__ == '__main__':
    # use gmm on cloud data
    K = 3
    BASE_COLOR = np.random.random((K, 3))
    X, _, _ = get_cloud_data(K, D=2)

    model = gmm(K)
    R, ll_history = model.fit(X)

    mixture_weight, mu, sigma = model.pi, model.mu, model.sigma

    plt.plot(ll_history)
    plt.title("Log-Likelihood")
    plt.show()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()

    print("pi:", mixture_weight)
    print("means:", mu)
    print("covariances:", sigma)

    X_test = np.array([
        [0, 0],
        [4, 4],
        [0, 4],
        [5, 5]])
    r_pred, y_pred = model.predict(X_test)
    print('predicted classes:', y_pred)

    # use gmm (sklearn implementation) on MNIST
    Kmnist = 10
    Nmnist = 10000
    Xmnist, _, Ymnist, _, picture_shape = get_mnist_data(train_size=0.8, should_plot_examples=False)
    Xmnist = Xmnist[:Nmnist]
    Ymnist = Ymnist[:Nmnist]

    model = GaussianMixture(n_components=10)
    model.fit(Xmnist)
    Mmnist = model.means_
    Rmnist = model.predict_proba(Xmnist)

    print("Purity:", purity_soft_cost(Ymnist, Rmnist)) # max is 1, higher is better
    print("DBI:", davis_bouldin_index_soft_cost(Xmnist, Rmnist, Mmnist)) # lower is better
