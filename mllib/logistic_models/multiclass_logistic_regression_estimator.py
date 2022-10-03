import numpy as np
import matplotlib.pyplot as plt
from scores import accuracy
from scores import multiclass_cross_entropy
from activations import softmax
from utils_ndarray import ndarray_one_hot_encode


class MulticlassLogisticRegression:
    def __init__(self, n_steps, n_classes=None, learning_rate=0.001, plot_training_history=False):
        self._activation = softmax
        self._n_steps = n_steps
        self._K = n_classes
        self._D = None
        self._learning_rate = learning_rate
        self._fit_history = None
        self._plot_training_history = plot_training_history

        self._target_label_size = None
        self.weights = None
        self.biases = None

    def fit(self, X, Y):
        assert isinstance(X, np.ndarray) and len(X.shape) == 2
        assert isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]

        if len(Y.shape) == 1:
            print('Assume targets is 1d array with values corresponding to K classes')
            Y_Kd = ndarray_one_hot_encode(Y, self._K)
            self._target_label_size = 1
        else:
            print('Assume targets is one hot encoded Kd array with columns corresponding to K classes')
            assert np.all(Y.sum(axis=1) == np.ones(Y.shape[0]))
            Y_Kd = Y
            self._target_label_size = Y_Kd.shape[1]

        self._D = X.shape[1]
        if self._K is None:
            self._K = len(set(Y))

        w0, b0 = self._initialize_weights()
        self.weights, self.biases, self._fit_history = self._backward(X, Y_Kd, self._n_steps, w0, b0,
                                                                      self._learning_rate)

    def predict(self, X):
        assert isinstance(X, np.ndarray)
        p_hat = self._forward(X, self.weights, self.biases)
        y_hat_ind = np.argmax(p_hat, axis=1)
        if self._target_label_size == 1:
            return y_hat_ind
        else:
            return ndarray_one_hot_encode(y_hat_ind, self._target_label_size)

    def score(self, X, Y):
        assert isinstance(Y, np.ndarray)
        if self._target_label_size == 1:
            assert len(Y.shape) == 1
        else:
            assert len(Y.shape) == 2
            assert Y.shape[1] == self._target_label_size
        y_hat = self.predict(X)
        return accuracy(Y, y_hat)

    def _initialize_weights(self):
        w0 = np.random.randn(self._D, self._K) / np.sqrt(self._D)
        b0 = np.random.randn(self._K)
        return w0, b0

    def _backward(self, x, y, n_steps, w0, b0, learning_rate):
        hist = []

        p_hat = self._forward(x, w0, b0)
        loss_start = multiclass_cross_entropy(y, p_hat)
        acc_start = accuracy(np.argmax(y, axis=1), np.argmax(p_hat, axis=1))
        hist.append(np.array([loss_start, acc_start]))

        w = w0
        b = b0
        for step in range(1, n_steps + 1):
            d_w = self._dw(x, y, p_hat)
            d_b = self._db(y, p_hat)
            w = w + learning_rate * d_w
            b = b + learning_rate * d_b

            p_hat = self._forward(x, w, b)
            loss = multiclass_cross_entropy(y, p_hat)
            acc = accuracy(np.argmax(y, axis=1), np.argmax(p_hat, axis=1))
            hist.append(np.array([loss, acc]))

        hist = np.array(hist)
        if self._plot_training_history:
            plt.figure(figsize=(10, 5))
            plt.plot(-1 * hist[:, 0], label='loss')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(hist[:, 1], label='accuracy')
            plt.legend()
            plt.show()
            print(f'final cost={hist[-1, 0]}, final accuracy={hist[-1, 1]}')

        return w, b, hist

    def _forward(self, x, w, b):
        assert isinstance(x, np.ndarray)
        return self._activation(x.dot(w) + b)

    def _dw(self, x, y, p_hat):
        return x.T.dot(y - p_hat)

    def _db(self, y, p_hat):
        return np.sum(y - p_hat, axis=0)
