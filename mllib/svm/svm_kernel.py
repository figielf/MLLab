import numpy as np
from matplotlib import pyplot as plt


class svm_kernel_gradient_descent:
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.w = None
        self.b = None
        self.x_fit = None
        self.y_fit = None

    def fit(self, x, y, learning_rate=1e-5, n_epochs=400, alpha0=None):
        N, D = x.shape
        self.x_fit = x
        self.y_fit = y
        self.w = np.random.randn(D)
        self.b = 0

        if alpha0 is not None:
            self.alpha = alpha0.copy()
        else:
            self.alpha = np.random.random(N)
        yyk = np.outer(y, y) * self.kernel(x, x)

        history = []
        for i in range(n_epochs):
            dalpha = -1 + yyk.dot(self.alpha)

            # we want to maximize cost so we use gradient ascent
            self.alpha = self.alpha - learning_rate * dalpha
            self.alpha[self.alpha < 0] = 0
            self.alpha[self.alpha > self.C] = self.C

            support_filter = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            b = y[support_filter] - (self.alpha * y).dot(self.kernel(x, x[support_filter]))
            self.b = b.mean()

            c = self.cost(yyk)
            print(f'linear svm - epoch: {i}, cost: {c}')
            history.append(c)
        print(f'loss:', history)
        return history

    def predict(self, x):
        return np.sign(self._decision_function(x))

    def _decision_function(self, x):
        return (self.alpha * self.y_fit).dot(self.kernel(self.x_fit, x)) + self.b

    def cost(self, yyk):
        return - self.alpha.sum() + 0.5 * self.alpha.reshape(1, -1).dot(yyk).dot(self.alpha.reshape(-1, 1))[0][0]

    def score(self, x, y):
        p = self.predict(x)
        return np.mean(y == p)


class svm_kernel_gradient_descent_on_primal:
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.u = None
        self.x_fit = None
        self.y_fit = None

    def fit(self, x, y, learning_rate=1e-5, n_epochs=400, u0=None, margins_hist=False):
        N, D = x.shape
        self.x_fit = x
        self.y_fit = y
        self.b = 0
        if u0 is not None:
            self.u = u0.copy()
        else:
            self.u = np.random.random(N)

        k = self.kernel(x, x)

        history = []
        for i in range(n_epochs):
            eta_filter = self._functional_margin(x, y) < 1
            du = k.dot(self.u) - self.C * y[eta_filter].dot(k[eta_filter])
            db = -self.C * np.sum(y[eta_filter])

            self.u = self.u - learning_rate * du
            self.b = self.b - learning_rate * db

            c = self.cost(x, y, k)
            print(f'linear svm - epoch: {i}, cost: {c}')
            history.append(c)

        self.support_ = np.where((y * self._decision_function(x)) <= 1)[0]
        print("num SVs:", len(self.support_))

        if margins_hist:
            # hist of margins
            m = y * (self.u.dot(k) + self.b)
            plt.hist(m, bins=20)
            plt.show()

        return history

    def predict(self, x):
        return np.sign(self._decision_function(x))

    def _decision_function(self, x):
        return self.u.dot(self.kernel(self.x_fit, x)) + self.b

    def _functional_margin(self, x, y):
        return y * (self._decision_function(x))

    def cost(self, x, y, k):
        return 0.5 * self.u.dot(k).dot(self.u) + self.C * np.maximum(0, 1 - self._functional_margin(x, y)).sum()

    def score(self, x, y):
        p = self.predict(x)
        return np.mean(y == p)
