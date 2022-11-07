import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from svm.svm_kernel import svm_kernel_gradient_descent_on_primal


def get_spiral():
    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0
        end_angle = start_angle + np.pi / 2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()

    X += np.random.randn(600, 2) * 0.5
    Y = np.array([0] * 100 + [1] * 100 + [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100)
    return X, Y


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2  # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    Y = np.array([0] * 100 + [1] * 100)
    return X, Y


def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    return X, Y


def get_clouds():
    N = 1000
    c1 = np.array([2, 2])
    c2 = np.array([-2, -2])
    # c1 = np.array([0, 3])
    # c2 = np.array([0, 0])
    X1 = np.random.randn(N, 2) + c1
    X2 = np.random.randn(N, 2) + c2
    X = np.vstack((X1, X2))
    Y = np.array([-1] * N + [1] * N)
    return X, Y


def clouds():
    X, Y = get_clouds()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, linear, 1e-5, 500


def medical():
    data = load_breast_cancer()
    X, Y = data.data, data.target
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, linear, 1e-3, 200


def xor():
    X, Y = get_xor()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=3.)
    return Xtrain, Xtest, Ytrain, Ytest, kernel, 1e-3, 500


def donut():
    X, Y = get_donut()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=1.)
    return Xtrain, Xtest, Ytrain, Ytest, kernel, 1e-3, 300


def spiral():
    X, Y = get_spiral()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=5.)
    return Xtrain, Xtest, Ytrain, Ytest, kernel, 1e-3, 500


# kernels
def linear(X1, X2, c=0):
    return X1.dot(X2.T) + c


def rbf(X1, X2, gamma=None):
    if gamma is None:
        gamma = 1.0 / X1.shape[-1]  # 1 / D
    # gamma = 0.05
    # gamma = 5. # for donut and spiral
    if np.ndim(X1) == 1 and np.ndim(X2) == 1:
        result = np.exp(-gamma * np.linalg.norm(X1 - X2) ** 2)
    elif (np.ndim(X1) > 1 and np.ndim(X2) == 1) or (np.ndim(X1) == 1 and np.ndim(X2) > 1):
        result = np.exp(-gamma * np.linalg.norm(X1 - X2, axis=1) ** 2)
    elif np.ndim(X1) > 1 and np.ndim(X2) > 1:
        result = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)
    return result


def sigmoid(X1, X2, gamma=0.05, c=1):
    return np.tanh(gamma * X1.dot(X2.T) + c)


def plot_decision_boundary(model, X, Y, resolution=100, colors=('b', 'k', 'r')):
    np.warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    grid = [[model._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and
    # make a scatter plot of training data
    ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
               linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(X[:, 0], X[:, 1],
               c=Y, lw=0, alpha=0.3, cmap='seismic')

    # Plot support vectors (non-zero alphas)
    # as circled points (linewidth > 0)
    mask = model.support_
    ax.scatter(X[:, 0][mask], X[:, 1][mask],
               c=Y[mask], cmap='seismic')

    # debug
    ax.scatter([0], [0], c='black', marker='x')

    # debug
    # x_axis = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    # w = model.w
    # b = model.b
    # # w[0]*x + w[1]*y + b = 0
    # y_axis = -(w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, y_axis, color='purple')
    # margin_p = (1 - w[0]*x_axis - b)/w[1]
    # plt.plot(x_axis, margin_p, color='orange')
    # margin_n = -(1 + w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, margin_n, color='orange')

    plt.show()


def run_model(Xtrain, Xtest, Ytrain, Ytest, lr, n_iters, model, desc, u0=None):
    t0 = datetime.now()
    history = model.fit(Xtrain, Ytrain, learning_rate=lr, n_epochs=n_iters, u0=u0)
    print("train duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("train score:", model.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("test score:", model.score(Xtest, Ytest), "duration:", datetime.now() - t0)

    plt.plot(history)
    plt.title(f'{desc} - loss per iteration')
    plt.show()

    if Xtrain.shape[1] == 2:
        plot_decision_boundary(model, Xtrain, Ytrain)


class KernelSVM:
    def __init__(self, kernel=linear, C=1.0):
        self.C = C
        self.kernel = kernel

    def _objective(self, margins):
        return 0.5 * self.u.dot(self.K.dot(self.u)) + \
               self.C * np.maximum(0, 1 - margins).sum()

    def fit(self, X, Y, learning_rate=1e-5, n_epochs=400, u0=None):
        N, D = X.shape
        self.N = N
        self.b = 0
        if u0 is not None:
            self.u = u0.copy()
        else:
            self.u = np.random.random(N)

        # setup kernel matrix
        self.X = X
        self.Y = Y
        self.K = self.kernel(X, X)

        # gradient descent
        losses = []
        for i, _ in enumerate(range(n_epochs)):
            margins = Y * (self.u.dot(self.K) + self.b)
            loss = self._objective(margins)
            losses.append(loss)

            idx = np.where(margins < 1)[0]
            grad_u = self.K.dot(self.u) - self.C * Y[idx].dot(self.K[idx])
            self.u -= learning_rate * grad_u
            grad_b = -self.C * Y[idx].sum()
            self.b -= learning_rate * grad_b

            print(f'linear svm - epoch: {i}, cost: {loss}')

        self.support_ = np.where((Y * (self.u.dot(self.K) + self.b)) <= 1)[0]
        print("num SVs:", len(self.support_))

        # print("w:", self.w)
        # print("b:", self.b)

        # hist of margins
        m = Y * (self.u.dot(self.K) + self.b)
        plt.hist(m, bins=20)
        plt.show()

        return losses

    def _decision_function(self, X):
        return self.u.dot(self.kernel(self.X, X)) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters = donut()
    print("Possible labels:", set(Ytrain))

    # make sure the targets are (-1, +1)
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1

    # scale the data - mandatory step in SVM
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    model1 = svm_kernel_gradient_descent_on_primal(kernel=kernel, C=1.0)
    run_model(Xtrain, Xtest, Ytrain, Ytest, lr, n_iters, model1, 'svm_kernel_gradient_descent_on_primal')
