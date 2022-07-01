from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from clustering.clustering_evaluation import purity_soft_cost
from tests.utils.data_utils import get_xor_data, get_donut_data, get_cloud_3d_data, get_mnist_data


def tsne_on_xor_2d():
    X, Y = get_xor_data(N=400, should_plot_data=False)
    model = TSNE(perplexity=40)
    X_tsne = model.fit_transform(X, Y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    ax1.set_title('oryginal data')
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, s=100, alpha=0.5)
    ax2.set_title('data reduced to 2d by tSNE')
    fig.suptitle('tSNE reduction results')
    plt.show()


def tsne_on_donut_2d():
    X, Y = get_donut_data(N=600, should_plot_data=False)
    model = TSNE(perplexity=40)
    X_tsne = model.fit_transform(X, Y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    ax1.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    ax1.set_title('oryginal data')
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, s=100, alpha=0.5)
    ax2.set_title('data reduced to 2d by tSNE')
    fig.suptitle('tSNE reduction results')
    plt.show()


def tsne_on_cloud_3d():
    X, Y = get_cloud_3d_data(100, should_plot_data=False)
    model = TSNE(perplexity=40)
    X_tsne = model.fit_transform(X, Y)

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], s=100, c=Y, alpha=0.5)
    ax1.set_title('oryginal data')
    ax2 = fig.add_subplot(122)
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, s=100, alpha=0.5)
    ax2.set_title('data reduced to 2d by tSNE')
    fig.suptitle('tSNE reduction results')
    plt.show()


def tsne_on_mnist_data():
    K = 10
    N = 1000
    Xtrain, _, Ytrain, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain = Xtrain[:N] / 255.0
    Ytrain = Ytrain[:N]

    tsne = TSNE()
    Z = tsne.fit_transform(Xtrain)
    plt.figure(figsize=(8, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=100, c=Ytrain, alpha=0.5)
    plt.show()

    # lets check purity measure
    # maximum purity is 1, higher is better
    gmm = GaussianMixture(n_components=10)
    gmm.fit(Xtrain)
    Rfull = gmm.predict_proba(Xtrain)
    print("Rfull.shape:", Rfull.shape)
    print("full purity without tSNE reduction:", purity_soft_cost(Ytrain, Rfull))

    # now try the same thing on the reduced data
    gmm2 = GaussianMixture(n_components=10)
    gmm2.fit(Z)
    Rreduced2 = gmm2.predict_proba(Z)
    print("reduced purity with tSNE reduction on whole tSNE output matrix:", purity_soft_cost(Ytrain, Rreduced2))

    gmm3 = GaussianMixture(n_components=10)
    gmm3.fit(Z[:, :2])
    Rreduced3 = gmm3.predict_proba(Z[:, :2])
    print("reduced purity with tSNE reduction on first two columns:", purity_soft_cost(Ytrain, Rreduced3))

    pca = PCA()
    Zpca = pca.fit_transform(Xtrain)
    gmm4 = GaussianMixture(n_components=10)
    gmm4.fit(Zpca)
    Rreduced4 = gmm4.predict_proba(Zpca)
    print("reduced purity with tSNE reduction on whole PCA output matrix:", purity_soft_cost(Ytrain, Rreduced4))

    gmm5 = GaussianMixture(n_components=10)
    gmm5.fit(Zpca[:, :2])
    Rreduced5 = gmm5.predict_proba(Zpca[:, :2])
    print("reduced purity with tSNE reduction on first two columns:", purity_soft_cost(Ytrain, Rreduced5))


if __name__ == '__main__':
    #tsne_on_xor_2d()
    #tsne_on_donut_2d()
    #tsne_on_cloud_3d()
    tsne_on_mnist_data()
