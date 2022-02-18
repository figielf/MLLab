import numpy as np
from matplotlib import pyplot as plt

from integration_tests.utils.data_utils import get_simple_xor_data, get_donut_data
from natural_networks.nn_activations import SigmoidActivation, SoftmaxActivation, TanhActivation, ReLUActivation
from natural_networks.nn_layers import DenseLayer
from natural_networks.nn_sequential_estimator import NNSequentialClassifier
from utils_ndarray import ndarray_one_hot_encode


def get_simple_cloud_data(Nclass):
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X4 = np.random.randn(Nclass, 2) + np.array([0, 4])
    X = np.vstack([X1, X2, X3, X4])

    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass + [3] * Nclass)
    return X, Y


def run_classification_cloud(Xtrain, Xtest, Ytrain, Ytest, plot_training_history=True):
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, s=100, alpha=0.5)
    plt.show()

    # randomly initialize weights
    D = 2  # dimensionality of input
    M1 = 3  # hidden layer size
    M2 = 5  # hidden layer size
    K = 4  # number of classes

    Ttrain = ndarray_one_hot_encode(Ytrain, K)
    Ttest = ndarray_one_hot_encode(Ytest, K)

    # 2 layers NN classifier
    W1 = np.random.randn(D, M1)
    b1 = np.random.randn(M1)
    W2 = np.random.randn(M1, K)
    b2 = np.random.randn(K)
    print(f'generated w0={W1}')
    print(f'generated b0={b1}')
    print(f'generated w1={W2}')
    print(f'generated b1={b2}')

    l0 = DenseLayer(D, M1, SigmoidActivation(), init_params=[W1, b1])
    l1 = DenseLayer(M1, K, SoftmaxActivation(), init_params=[W2, b2])

    model2l = NNSequentialClassifier(layers=[l0, l1], n_steps=100, n_classes=K,
                                     plot_training_history=plot_training_history)

    model2l.fit(Xtrain, Ytrain)
    assert np.all(model2l.score(Xtrain, model2l.predict(Xtrain))) == 1

    print('\n')
    print(f'final 2 layers model w0={model2l.layers[0].weights}')
    print(f'final 2 layers model b0={model2l.layers[0].biases}')
    print(f'final 2 layers model w1={model2l.layers[1].weights}')
    print(f'final 2 layers model b1={model2l.layers[1].biases}')
    print('\n')
    print('Train accuracy in 2 layers nn:', model2l.score(Xtrain, Ytrain))
    print('Test accuracy in 2 layers nn:', model2l.score(Xtest, Ytest))

    # 3 layers NN classifier
    # W2_2 = np.random.randn(M1, M2)
    # b2_2 = np.random.randn(M2)
    # W3_2 = np.random.randn(M2, K)
    # b3_2 = np.random.randn(K)
    l0 = DenseLayer(D, M1, SigmoidActivation())  # , init_params=[W1, b1])
    l1 = DenseLayer(M1, M2, ReLUActivation())  # , init_params=[W2_2, b2_2])
    l2 = DenseLayer(M2, K, SoftmaxActivation())  # , init_params=[W3_2, b3_2])
    model3l = NNSequentialClassifier(layers=[l0, l1, l2], learning_rate=0.0001, n_steps=1000, n_classes=K,
                                     plot_training_history=plot_training_history)

    model3l.fit(Xtrain, Ttrain)
    assert np.all(model3l.score(Xtrain, model3l.predict(Xtrain))) == 1

    print('\n')
    print(f'final 3 layers model w0={model3l.layers[0].weights}')
    print(f'final 3 layers model b0={model3l.layers[0].biases}')
    print(f'final 3 layers model w1={model3l.layers[1].weights}')
    print(f'final 3 layers model b1={model3l.layers[1].biases}')
    print(f'final 3 layers model w2={model3l.layers[2].weights}')
    print(f'final 3 layers model b2={model3l.layers[2].biases}')
    print('\n')
    print('Train accuracy in 3 layers nn:', model3l.score(Xtrain, Ttrain))
    print('Test accuracy in 3 layers nn:', model3l.score(Xtest, Ttest))


def run_classification_xor(X, Y, plot_training_history=True):
    D = 2
    M = 5
    M1 = 4
    M2 = 3
    K = 2
    # W1 = np.random.randn(D, M)
    # b1 = np.zeros(M)
    # W2 = np.random.randn(M, K)
    # b2 = np.zeros(K)
    # print(f'generated w0={W1}')
    # print(f'generated b0={b1}')
    # print(f'generated w1={W2}')
    # print(f'generated b1={b2}')

    l0 = DenseLayer(D, M, ReLUActivation())  # , init_params=[W1, b1])
    l1 = DenseLayer(M, M1, SigmoidActivation())  # , init_params=[W2, b2])
    l2 = DenseLayer(M1, M2, ReLUActivation())  # , init_params=[W2, b2])
    lN = DenseLayer(M2, K, SoftmaxActivation())  # , init_params=[W2, b2])

    xor_model = NNSequentialClassifier(layers=[l0, l1, l2, lN], learning_rate=1e-2, n_steps=2, n_classes=K,
                                       plot_training_history=plot_training_history)

    xor_model.fit(X, Y)
    assert np.all(xor_model.score(X, xor_model.predict(X))) == 1

    print('\n')
    print(f'final xor model w0={xor_model.layers[0].weights}')
    print(f'final xor model b0={xor_model.layers[0].biases}')
    print(f'final xor model w1={xor_model.layers[1].weights}')
    print(f'final xor model b1={xor_model.layers[1].biases}')
    print('\n')
    print('Train accuracy in xor:', xor_model.score(X, Y))


def run_classification_donut(X, Y, plot_training_history=True):
    D = 2
    M = 8
    K = 2
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)
    print(f'generated w0={W1}')
    print(f'generated b0={b1}')
    print(f'generated w1={W2}')
    print(f'generated b1={b2}')

    l0 = DenseLayer(D, M, ReLUActivation(), init_params=[W1, b1])
    l1 = DenseLayer(M, K, SoftmaxActivation(), init_params=[W2, b2])

    model = NNSequentialClassifier(layers=[l0, l1], learning_rate=0.00001, n_steps=30000, n_classes=K,
                                   plot_training_history=plot_training_history)

    model.fit(X, Y)
    assert np.all(model.score(X, model.predict(X))) == 1

    print('\n')
    print(f'final xor model w0={model.layers[0].weights}')
    print(f'final xor model b0={model.layers[0].biases}')
    print(f'final xor model w1={model.layers[1].weights}')
    print(f'final xor model b1={model.layers[1].biases}')
    print('\n')
    print('Train accuracy in xor:', model.score(X, Y))


if __name__ == '__main__':
    Xxor, Yxor = get_simple_xor_data(should_plot_data=False)
    run_classification_xor(Xxor, Yxor, plot_training_history=False)

    Xdonut, Ydonut = get_donut_data(100, should_plot_data=False)
    run_classification_donut(Xdonut, Ydonut, plot_training_history=False)

    Xcloud_train, Ycloud_train = get_simple_cloud_data(500)
    Xcloud_test, Ycloud_test = get_simple_cloud_data(100)
    run_classification_cloud(Xcloud_train, Xcloud_test, Ycloud_train, Ycloud_test, plot_training_history=False)
