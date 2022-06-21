import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from dimensionality_reduction.pca import pca
from integration_tests.utils.data_utils import get_mnist_data


def reduce_by_sklearn_PCA(X):
    model = PCA()
    reduced = model.fit_transform(X)
    return reduced, model.explained_variance_ratio_


def reduce_by_my_PCA(X):
    model = pca()
    reduced = model.fit_transform(X)
    return reduced, model.explained_variance_ratio


def print_pca_results(X, Y, plot_figures=True):
    sklearn_reduced, sklearn_explained_variance_ratio = reduce_by_sklearn_PCA(X)
    my_reduced, my_explained_variance_ratio = reduce_by_my_PCA(X)

    if plot_figures == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.scatter(sklearn_reduced[:, 0], sklearn_reduced[:, 1], s=100, c=Y, alpha=0.5)
        ax1.set_title('sklearn PCA')
        ax2.scatter(my_reduced[:, 0], my_reduced[:, 1], s=100, c=Y, alpha=0.5)
        ax2.set_title('my PCA')
        fig.suptitle('X reduced to first two dimensions')
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.plot(sklearn_explained_variance_ratio)
        ax1.set_title('sklearn PCA')
        ax2.plot(my_explained_variance_ratio)
        ax2.set_title('my PCA')
        fig.suptitle('explained_variance_ratio')
        plt.show()

        # cumulative variance
        # choose k = number of dimensions that gives us 95-99% variance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.plot(np.cumsum(sklearn_explained_variance_ratio))
        ax1.set_title('sklearn PCA')
        ax2.plot(np.cumsum(my_explained_variance_ratio))
        ax2.set_title('my PCA')
        fig.suptitle('cumulative explained_variance_ratio')
        plt.show()


if __name__ == '__main__':
    K = 10
    N = 1000
    Xtrain, _, Ytrain, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain = Xtrain[:N]
    Ytrain = Ytrain[:N]

    print_pca_results(Xtrain, Ytrain, plot_figures=True)
