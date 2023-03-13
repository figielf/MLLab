import numpy as np


class sgd_with_momentum_linear_model:
    def __init__(self, input_dim, n_classes):
        self.D = input_dim
        self.K = n_classes

        self.W = np.random.randn(self.D, self.K) / np.sqrt(self.D)
        self.b = np.zeros(self.K)

        # momentum params
        self.vW = np.zeros((self.D, self.K))
        self.vb = np.zeros(self.K)

        self.history = {'mse': []}

    def fit_one_batch(self, X, Y, learning_rate=0.01, momentum=0.9):
        assert len(X.shape) == 2
        assert X.shape[1] == self.D
        assert len(Y.shape) == 2
        assert Y.shape[1] == self.K

        n_points = len(Y) * self.K
        error = self.predict(X) - Y
        dW = 2 * X.T.dot(error) / n_points  # calc exact gradient, not adjusting the learning rate
        db = 2 * error.sum(axis=0) / n_points

        self.vW = momentum * self.vW - learning_rate * dW
        self.vb = momentum * self.vb - learning_rate * db

        self.W = self.W + self.vW
        self.b = self.b + self.vb

        mse = np.mean(error ** 2)
        self.history['mse'].append(mse)
        return mse

    def predict(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.D

        return X.dot(self.W) + self.b

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)
