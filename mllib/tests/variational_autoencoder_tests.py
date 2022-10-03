import numpy as np
from matplotlib import pyplot as plt

from tests.utils.data_utils import get_mnist_data
from autoencoders.variational_autoencoder import variational_autoencoder_for_binary_variable


def plot_generated_samples(vae_model, X, Y):
    plt.figure(figsize=(16, 16))
    for k in range(10):
        i = np.random.choice(len(X))
        x = X[i]
        im = vae_model.posterior_predictive_sample([x]).reshape(picture_shape)
        plt.subplot(5, 4, 2 * k + 1)
        plt.imshow(x.reshape(picture_shape), cmap='gray')
        plt.title(f'Original digit {Y[i]}')
        plt.subplot(5, 4, 2 * k + 2)
        plt.imshow(im, cmap='gray')
        plt.title(f'Sampled digit {Y[i]}')
    plt.title('Posterior predictive sampling')
    plt.show()

    plt.figure(figsize=(16, 16))
    for k in range(10):
        im, probs = vae_model.prior_predictive_sample_with_probs()
        im = im.reshape(picture_shape)
        probs = probs.reshape(picture_shape)
        plt.subplot(5, 4, 2 * k + 1)
        plt.imshow(im, cmap='gray')
        plt.title(f'Prior predictive sample')
        plt.subplot(5, 4, 2 * k + 2)
        plt.imshow(probs, cmap='gray')
        plt.title(f'Prior predictive probs')
    plt.title('Prior predictive sampling')
    plt.show()


def visualize_latent_space(X, Y):
    print('\nTraining variational autoencoder with layer sizes: [200, 100, 2]')
    vae = variational_autoencoder_for_binary_variable(784, [200, 100, 2])
    vae.fit(X.copy(), learning_rate=0.0001)

    Z = vae.transform(X)
    plt.scatter(Z[:, 0], Z[:, 1], c=Y, s=10)
    plt.show()

    n = 20
    x_values = np.linspace(-3, 3, n)
    y_values = np.linspace(-3, 3, n)
    image = np.empty((28 * n, 28 * n))

    Z_grid = []
    for x in x_values:
        for y in y_values:
            z = [x, y]
            Z_grid.append(z)
    X_reconstructed = vae.prior_predictive_with_input(Z_grid)

    k = 0
    for i in range(n):
        for j in range(n):
            x_hat = X_reconstructed[k].reshape(28, 28)
            image[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_hat
            k += 1
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    K = 10
    N = 1000
    X, _, Y, _, picture_shape = get_mnist_data(train_size=1.0, should_plot_examples=False)
    #X = X[:N]
    #Y = Y[:N]
    # convert X to binary variable
    X = (X > 0.5).astype(np.float32)

    vae = variational_autoencoder_for_binary_variable(784, [200, 100])
    print('\nTraining variational autoencoder with layer sizes: [200, 100]')
    costs = vae.fit(X.copy(), learning_rate=0.0001)
    plt.plot(costs)
    plt.show()

    plot_generated_samples(vae, X, Y)

    visualize_latent_space(X, Y)
