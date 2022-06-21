import numpy as np
from matplotlib import pyplot as plt

from integration_tests.utils.data_utils import get_mnist_data

if __name__ == '__main__':
    K = 10
    test_size = 1000
    Xtrain, Xtest, Ytrain, Ytest, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain, Ytrain = Xtrain[:-test_size], Ytrain[:-test_size]
    Xtest, Ytest = Xtest[-test_size:], Ytest[-test_size:]

    autoencoder_model = autoencoder(300, 0)
    autoencoder_model.fit(Xtrain, epochs=2, show_fig=True)

    done = False
    while not done:
        i = np.random.choice(len(Xtest))
        x = Xtest[i]
        y = autoencoder_model.predict([x])
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('original')

        plt.subplot(1, 2, 2)
        plt.imshow(y.reshape(28, 28), cmap='gray')
        plt.title('reconstructed')

        plt.show()

        ans = input("generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True