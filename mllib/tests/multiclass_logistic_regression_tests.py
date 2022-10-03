import numpy as np
from matplotlib import pyplot as plt

from logistic_models.multiclass_logistic_regression_estimator import MulticlassLogisticRegression
from utils_ndarray import ndarray_one_hot_encode


def det_simple_cloud_data(Nclass):
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X4 = np.random.randn(Nclass, 2) + np.array([0, 4])
    X = np.vstack([X1, X2, X3, X4])

    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass + [3] * Nclass)
    return X, Y


if __name__ == '__main__':
    Xtrain, Ytrain = det_simple_cloud_data(500)
    Xtest, Ytest = det_simple_cloud_data(100)

    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, s=100, alpha=0.5)
    plt.show()

    D = 2  # dimensionality of input
    M1 = 3  # hidden layer size
    M2 = 5  # hidden layer size
    K = 4  # number of classes

    Ttrain = ndarray_one_hot_encode(Ytrain, K)
    Ttest = ndarray_one_hot_encode(Ytest, K)

    model = MulticlassLogisticRegression(n_steps=100, n_classes=K, learning_rate=0.0001, plot_training_history=True)
    model.fit(Xtrain, Ytrain)

    assert model.score(Xtrain, model.predict(Xtrain)) == 1

    print('Train accuracy:', model.score(Xtrain, Ytrain))
    print('Test accuracy:', model.score(Xtest, Ytest))
