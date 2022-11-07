import numpy as np
from matplotlib import pyplot as plt


class svm_linear:
    def __init__(self, C):
        self.C = C
        self.w = None
        self.b = None

    def fit(self, x, y, learning_rate=0.001, n_epochs=10, margins_hist=False):
        D = x.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        history = []
        for i in range(n_epochs):
            eta_filter = self._functional_margin(x, y) < 1
            dw = self.w - self.C * y[eta_filter].dot(x[eta_filter])
            db = -self.C * np.sum(y[eta_filter])

            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

            c = self.cost(x, y)
            print(f'linear svm - epoch: {i}, cost: {c}')
            history.append(c)

        self.support_ = np.where((y * self._x_prod_w(x)) <= 1)[0]
        print("num SVs:", len(self.support_))

        if margins_hist:
            # hist of margins
            m = y * self._x_prod_w(x)
            plt.hist(m, bins=20)
            plt.show()

        return history

    def predict(self, x):
        return np.sign(self._x_prod_w(x))

    def _x_prod_w(self, x):
        return x.dot(self.w) + self.b

    def _functional_margin(self, x, y):
        return y * (self._x_prod_w(x))

    def cost(self, x, y):
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - self._functional_margin(x, y)).sum()

    def score(self, x, y):
        p = self.predict(x)
        return np.mean(y == p)

