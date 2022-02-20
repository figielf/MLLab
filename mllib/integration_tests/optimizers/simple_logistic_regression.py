import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from integration_tests.utils.data_utils import get_mnist_data
from scores import binary_cross_entropy, accuracy

RANDOM_STATE = 123

def _dW(X, errors):
    return X.T.dot(errors)


def _db(errors):
    return errors.sum()

def simple_logistic_regression(X, Y, W0, b0, n_epochs, learning_rate):
    print('W0[:5]:', W0[:5])
    print('b0:', b0)

    W = W0
    b = b0

    history = []
    for epoch in range(n_epochs):
        print(f'----------epoch {epoch}----------')
        y_hat = X.dot(W) + b

        errors = y_hat - Y

        W = W - learning_rate * _dW(X, errors)
        b = b - learning_rate * _db(errors)

        loss = binary_cross_entropy(Y, y_hat)
        acc = accuracy(Y, y_hat)
        err = 1 - acc
        history.append(np.array([loss, acc, err]))

    history = np.array(history)

    plt.figure(figsize=(20,10))
    plt.plot(history[:, 0], label='train loss')
    plt.plot(history[:, 1], label='train accuracy')
    plt.plot(history[:, 2], label='train error rate')
    plt.show()

    print('W[:5] trained:', W[:5])
    print('b trained:', b)
    print('W0[:5] check:', W0[:5])
    print('b0 check:', b0)
    return W, b, history


if __name__ == '__main__':
    X, Y, picture_shape = get_mnist_data(should_shuffle=False, should_plot_examples=False)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
    assert Xtrain.shape[1] == picture_shape[0] * picture_shape[1]

    print('Xtrain.shape:', Xtrain.shape)
    print('Ytrain.shape:', Ytrain.shape)
    print('Xtest.shape:', Xtest.shape)
    print('Ytest.shape:', Ytest.shape)

    D = Xtrain.shape[1]
    W0 = np.random.randn(D, 1) / np.sqrt(D)
    b0 = 0
    simple_logistic_regression(Xtrain, Ytrain, W0, b0, n_epochs=2, learning_rate=0.01)


