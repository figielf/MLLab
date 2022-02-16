import numpy as np
from matplotlib import pyplot as plt

from integration_tests.utils.data_utils import get_simple_xor_data, get_donut_data
from natural_networks.nn_activations import SigmoidActivation, SoftmaxActivation, TanhActivation, ReLUActivation, \
    IdentityActivation
from natural_networks.nn_layers import DenseLayer
from natural_networks.nn_sequential_estimator import NNSequentialClassifier, NNSequentialRegressor
from utils_ndarray import ndarray_one_hot_encode


def get_simple_saddle_data(N):
    X = np.random.random((N, 2)) * 4 - 2  # in between (-2, +2)
    Y = X[:, 0] * X[:, 1]  # makes a saddle shape
    return X, Y


def run_regression_saddle(Xtrain, Xtest, Ytrain, Ytest, plot_training_history=True):
    D = 2
    M = 3
    K = 1

    # 2 layers NN regressor
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, 1) / np.sqrt(M)
    b2 = 0
    l0 = DenseLayer(D, M, SigmoidActivation(), init_params=[W1, b1])
    lN = DenseLayer(M, K, IdentityActivation(), init_params=[W2, b2])
    model = NNSequentialRegressor(layers=[l0, lN], n_steps=2, learning_rate=1e-4, plot_training_history=plot_training_history)

    model.fit(Xtrain, Ytrain)
    assert model.score(Xtrain, model.predict(Xtrain)) == 0

    print('\n')
    print(f'final 3 layers model w0={model.layers[0].weights}')
    print(f'final 3 layers model b0={model.layers[0].biases}')
    print(f'final 3 layers model w1={model.layers[1].weights}')
    print(f'final 3 layers model b1={model.layers[1].biases}')
    print('\n')
    print('Train accuracy in 3 layers nn:', model.score(Xtrain, Ytrain))
    print('Test accuracy in 3 layers nn:', model.score(Xtest, Ytest))


if __name__ == '__main__':
    Xtrain, Ytrain = get_simple_saddle_data(100)
    Xtest, Ytest = get_simple_saddle_data(200)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain)
    #plt.show()

    run_regression_saddle(Xtrain, Xtest, Ytrain, Ytest, plot_training_history=False)
