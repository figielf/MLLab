import numpy as np
from matplotlib import pyplot as plt

from activations import softmax
from integration_tests.utils.data_utils import get_mnist_normalized_data
from scores import accuracy, multiclass_cross_entropy
from utils_ndarray import ndarray_one_hot_encode, one_hot_2_vec

RANDOM_STATE = 123

def _dW(X, errors):
    return X.T.dot(errors)


def _db(errors):
    return errors.sum(axis=0)

def _forward(X, W, b):
    return softmax(X.dot(W) + b)

def simple_logistic_regression(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, learning_rate, reg):
    print(f'simple_logistic_regression - W0.mean()={W0.mean()}, W0.std()={W0.std()}')
    print(f'simple_logistic_regression - b0.mean()={b0.mean()}, b0.std()={b0.std()}')
    W = W0
    b = b0

    history = []
    for epoch in range(n_epochs):
        #print(f'----------epoch {epoch}----------')
        p_hat = _forward(Xtrain, W, b)
        loss = multiclass_cross_entropy(Ytrain, p_hat)
        y_hat = np.argmax(p_hat, axis=1)
        p_hat_test = _forward(Xtest, W, b)
        loss_test = multiclass_cross_entropy(Ytest, p_hat_test)
        y_hat_test = np.argmax(p_hat_test, axis=1)

        acc = accuracy(one_hot_2_vec(Ytrain), y_hat)
        acc_test = accuracy(one_hot_2_vec(Ytest), y_hat_test)
        history.append(np.array([loss, loss_test, acc, acc_test]))

        errors = p_hat - Ytrain
        W = W - learning_rate * (_dW(Xtrain, errors) + reg * W)
        b = b - learning_rate * (_db(errors) + reg * b)
    history = np.array(history)

    print('Final train cost:', history[-1,0])
    print('Final test cost:', history[-1,1])
    print('Final train error rate:', 1 - history[-1,2])
    print('Final test error rate:', 1 - history[-1,3])

    plt.figure(figsize=(20,20))
    plt.subplot(2, 1, 1)
    plt.plot(history[:, 0], label='train loss')
    plt.plot(history[:, 1], label='test loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(1 - history[:, 2], label='train error rate')
    plt.plot(1 - history[:, 3], label='test error rate')
    plt.legend()
    plt.show()
    return W, b, history


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest, picture_shape = get_mnist_normalized_data(train_size=1000, should_plot_examples=False)
    assert Xtrain.shape[1] == picture_shape[0] * picture_shape[1]

    K = 10
    Ytrain = ndarray_one_hot_encode(Ytrain, K)
    Ytest = ndarray_one_hot_encode(Ytest, K)

    print('Xtrain.shape:', Xtrain.shape)
    print('Ytrain.shape:', Ytrain.shape)
    print('Xtest.shape:', Xtest.shape)
    print('Ytest.shape:', Ytest.shape)

    D = Xtrain.shape[1]
    W0 = np.random.randn(D, K) / np.sqrt(D)
    b0 = np.zeros(K)
    W, b, _ = simple_logistic_regression(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs=500, learning_rate=0.00003, reg=1)
    print(f'final - W.mean()={W.mean()}, W0.std()={W.std()}')
    print(f'final - b.mean()={b.mean()}, b0.std()={b.std()}')
