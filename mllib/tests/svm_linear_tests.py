from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from svm.svm_linear import svm_linear


def plot_decision_boundary(model, X, Y, resolution=100, colors=('b', 'k', 'r')):
    np.warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    grid = [[model._x_prod_w(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
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


def clouds():
    X, Y = get_clouds_data(1000)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200


def get_clouds_data(n):
  c1 = np.array([2, 2])
  c2 = np.array([-2, -2])
  # c1 = np.array([0, 3])
  # c2 = np.array([0, 0])
  X1 = np.random.randn(n, 2) + c1
  X2 = np.random.randn(n, 2) + c2
  X = np.vstack((X1, X2))
  Y = np.array([-1] * n + [1] * n)
  return X, Y


def medical():
    data = load_breast_cancer()
    X, Y = data.data, data.target
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest, lr, n_iters = clouds()
    #Xtrain, Xtest, Ytrain, Ytest, lr, n_iters = medical()
    print("Possible labels:", set(Ytrain))

    # make sure the targets are (-1, +1)
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1

    # scale the data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # now we'll use our custom implementation
    model = svm_linear(C=1.0)

    t0 = datetime.now()
    history = model.fit(Xtrain, Ytrain, learning_rate=lr, n_epochs=n_iters)
    print("train duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("train score:", model.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("test score:", model.score(Xtest, Ytest), "duration:", datetime.now() - t0)
    print("model.w:", model.w)
    print("model.b:", model.b)

    plt.plot(history)
    plt.title("loss per iteration")
    plt.show()

    if Xtrain.shape[1] == 2:
        plot_decision_boundary(model, Xtrain, Ytrain)
