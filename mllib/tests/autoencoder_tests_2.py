import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from autoencoders.deep_autoencoder import deep_autoencoder

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

from tests.utils.data_utils import get_mnist_data

if __name__ == '__main__':
    K = 10
    test_size = 1000
    Xtrain, _, Ytrain, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain, Ytrain = Xtrain[:-test_size], Ytrain[:-test_size]
    Xtest, Ytest = Xtrain[-test_size:], Ytrain[-test_size:]

    _, D = Xtest.shape
    autoencoder = deep_autoencoder([500, 300, 2])
    with tf.compat.v1.Session() as session:
        autoencoder.set_session(session)
        history = autoencoder.fit(Xtrain, n_epochs=30)
        plt.plot(history)
        plt.show()

        n_cases = 5
        plt.figure(figsize=(10, 16))
        for k in range(1, n_cases + 1):
            i = np.random.choice(len(Xtest))
            x = Xtest[i]
            y = autoencoder.predict(np.array([x]))
            plt.subplot(n_cases, 2, 2 * k - 1)
            plt.imshow(x.reshape(picture_shape), cmap='gray')
            plt.title('Original')

            plt.subplot(n_cases, 2, 2 * k)
            plt.imshow(y.reshape(picture_shape), cmap='gray')
            plt.title('Reconstructed')
        plt.show()
