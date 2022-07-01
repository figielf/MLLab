import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from clustering.clustering_evaluation import purity
from deep_autoencoder import deep_autoencoder
from integration_tests.utils.data_utils import get_mnist_data

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


if __name__ == '__main__':
    K = 10
    test_size = 1000
    Xtrain, _, Ytrain, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain, Ytrain = Xtrain[:-test_size], Ytrain[:-test_size]

    autoencoder = deep_autoencoder([500, 300, 2])
    with tf.compat.v1.Session() as session:
        autoencoder.set_session(session)
        history = autoencoder.fit(Xtrain.copy(), n_epochs=10)

        plt.figure(figsize=(10, 16))
        plt.subplot(2, 1, 1)
        plt.plot(history)
        plt.title('train history')

        mapping = autoencoder.map2center(Xtrain.copy())
        print(np.array(mapping).shape)

        plt.subplot(2, 1, 2)
        plt.scatter(mapping[:, 0], mapping[:, 1], c=Ytrain, s=100, alpha=0.5)
        plt.title('2D data representation')
        plt.show()

    # compare with gaussian mixture model
    gmm = GaussianMixture(n_components=10)
    gmm.fit(Xtrain.copy())
    print("Finished GMM training")
    responsibilities_full = gmm.predict_proba(Xtrain.copy().copy())
    print("full purity:", purity(Ytrain.copy(), responsibilities_full))

    gmm.fit(mapping)
    responsibilities_reduced = gmm.predict_proba(mapping)
    print("reduced purity:", purity(Ytrain.copy(), responsibilities_reduced))
