import numpy as np
import matplotlib.pyplot as plt
from bagging_estimator import BaggingClassifier
from integration_tests.utils.plot_utils import plot_decision_boundary_2d
from sklearn.tree import DecisionTreeClassifier


def plot_data(X, Y):
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plt.show()


def get_noisy_xor_data():
    sep = 2
    X = np.random.randn(N, D)
    X[:N // 4] += np.array([sep, sep])
    X[N // 4:2 * N // 4] += np.array([sep, -sep])
    X[2 * N // 4:3 * N // 4] += np.array([-sep, -sep])
    X[3 * N // 4:] += np.array([-sep, sep])
    Y = np.array([0] * (N // 4) + [1] * (N // 4) + [0] * (N // 4) + [1] * (N // 4))
    return X, Y


def get_noisy_circles_data():
    sep = 1.5
    X = np.random.randn(N, D)
    X[:N // 2] += np.array([sep, sep])
    X[N // 2:] += np.array([-sep, -sep])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    return X, Y


if __name__ == '__main__':
    np.random.seed(10)
    N = 500
    D = 2

    # noisy xor in single (not bagged) model for comparison
    X, Y = get_noisy_xor_data()

    model = DecisionTreeClassifier()
    model.fit(X, Y)
    print("score for 1 tree:", model.score(X, Y))

    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plot_decision_boundary_2d(X, model)
    plt.show()

    # noisy xor in bagging model
    model = BaggingClassifier(lambda: DecisionTreeClassifier(max_depth=2), 200)
    model.fit(X, Y)

    print("score for bagged model:", model.score(X, Y))

    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plot_decision_boundary_2d(X, model)
    plt.show()
