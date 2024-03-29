import numpy as np
from matplotlib import pyplot as plt
from natural_networks.nn_activations import SigmoidActivation, TanhActivation, ReLUActivation, \
    IdentityActivation
from natural_networks.nn_layers import DenseLayer
from natural_networks.nn_sequential_estimator import NNSequentialRegressor


def get_simple_saddle_data(N):
    X = np.random.random((N, 2)) * 4 - 2  # in between (-2, +2)
    Y = X[:, 0] * X[:, 1]  # makes a saddle shape
    return X, Y


def run_regression_saddle(Xtrain, Xtest, Ytrain, Ytest, plot_training_history=True):
    D = 2
    M = 100
    M1 = 100
    K = 1

    # 3 layers NN regressor
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, M1) / np.sqrt(M)
    b2 = np.random.randn(M1)
    Wn = np.random.randn(M1, 1) / np.sqrt(M)
    bn = 0
    l0 = DenseLayer(D, M, ReLUActivation(), init_params=[W1, b1])
    l1 = DenseLayer(M, M1, SigmoidActivation(), init_params=[W2, b2])
    ln = DenseLayer(M1, K, IdentityActivation(), init_params=[Wn, bn])
    model = NNSequentialRegressor(layers=[l0, l1, ln], n_steps=2000, learning_rate=1e-4,
                                  plot_training_history=plot_training_history)

    model.fit(Xtrain, Ytrain)
    assert model.score(Xtrain, model.predict(Xtrain)) == 0

    # print('\n')
    # print(f'final 3 layers model w0={model.layers[0].weights}')
    # print(f'final 3 layers model b0={model.layers[0].biases}')
    # print(f'final 3 layers model w1={model.layers[1].weights}')
    # print(f'final 3 layers model b1={model.layers[1].biases}')
    # print(f'final 3 layers model w2={model.layers[2].weights}')
    # print(f'final 3 layers model b2={model.layers[2].biases}')
    print('\n')
    print('Train MSE in 3 layers nn:', model.score(Xtrain, Ytrain))
    print('Test MSE in 3 layers nn:', model.score(Xtest, Ytest))


if __name__ == '__main__':
    Xtrain, Ytrain = get_simple_saddle_data(500)
    Xtest, Ytest = get_simple_saddle_data(100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xtrain[:, 0], Xtrain[:, 1], Ytrain)
    plt.show()

    run_regression_saddle(Xtrain, Xtest, Ytrain, Ytest, plot_training_history=True)
