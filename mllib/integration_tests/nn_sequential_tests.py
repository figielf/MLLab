import numpy as np
from matplotlib import pyplot as plt

from natural_networks.AnnClassifier_2l_oryginal import AnnClassifier_2l_oryginal
from natural_networks.AnnClassifier_2l_recursive import AnnClassifier_2l_recursive
from natural_networks.nn_activations import SigmoidActivation, SoftmaxActivation, TanhActivation, ReLUActivation
from natural_networks.nn_layers import DenseLayer
from natural_networks.nn_sequential_estimator import AnnClassifier_with_layers


def compare(modelclass1, modelclass2, n_steps=4, eps=1e-10, plot_training_history=True):
    model1 = modelclass1(n_steps=n_steps, n_neurons_hidden=M, n_classes=K, plot_training_history=plot_training_history)
    model1.fit(X, Y, [W1, W2], [b1, b2])

    model2 = modelclass2(n_steps=n_steps, n_neurons_hidden=M, n_classes=K, plot_training_history=plot_training_history)
    model2.fit(X, Y, [W1, W2], [b1, b2])

    assert len(model1.weights) == len(model2.weights)
    for layer in range(len(model1.weights)):
        print(f'layer-{layer} max weigths diff:', np.max(np.abs(model1.weights[layer] - model2.weights[layer])))
        assert np.max(np.abs(model1.weights[layer] - model2.weights[layer])) < eps

    assert len(model1.biases) == len(model2.biases)
    for layer in range(len(model1.biases)):
        print(f'layer-{layer} max biases diff:', np.max(np.abs(model1.biases[layer] - model2.biases[layer])))
        assert np.max(np.abs(model1.biases[layer] - model2.biases[layer])) < eps

    print(f'max fit history diff:', np.max(np.abs(model1._fit_history - model2._fit_history)))
    assert np.max(np.abs(model1._fit_history - model2._fit_history)) < eps

    print('ALL GOOD')

def compare2(modelclass1, modelclass2, layers, n_steps=4, eps=1e-10, plot_training_history=True):
    model1 = modelclass1(n_steps=n_steps, n_neurons_hidden=M1, n_classes=K, plot_training_history=plot_training_history)
    model1.fit(X, Y, [W1, W2], [b1, b2])

    model2 = modelclass2(layers=layers, n_steps=n_steps, n_classes=K, plot_training_history=plot_training_history)
    model2.fit(X, Y)

    print('\n')
    print(f'final model1 w0={model1.weights[0]}')
    print(f'final model1 b0={model1.biases[0]}')
    print(f'final model1 w1={model1.weights[1]}')
    print(f'final model1 b1={model1.biases[1]}')
    print('\n')
    print(f'final model2 w0={model2.layers[0].weights}')
    print(f'final model2 b0={model2.layers[0].biases}')
    print(f'final model2 w1={model2.layers[1].weights}')
    print(f'final model2 b1={model2.layers[1].biases}')
    print('\n')

    assert len(model1.weights) == len(model2.layers)
    for i in range(len(model1.weights)):
        layer = model2.layers[i]
        print(f'layer-{i} max weigths diff:', np.max(np.abs(model1.weights[i] - layer.weights)))
        assert np.max(np.abs(model1.weights[i] - layer.weights)) < eps

    assert len(model1.biases) == len(model2.layers)
    for layer in range(len(model1.biases)):
        layer = model2.layers[i]
        print(f'layer-{i} max biases diff:', np.max(np.abs(model1.biases[i] - layer.biases)))
        assert np.max(np.abs(model1.biases[i] - layer.biases)) < eps

    print(f'max fit history diff:', np.max(np.abs(model1._fit_history - model2._fit_history)))
    assert np.max(np.abs(model1._fit_history - model2._fit_history)) < eps

    print('ALL GOOD')


def compare3(modelclass2, layers, n_steps=4, plot_training_history=True):

    model2 = modelclass2(layers=layers, n_steps=n_steps, n_classes=K, plot_training_history=plot_training_history)
    model2.fit(X, Y)

    print('\n')
    print(f'final model2 w0={model2.layers[0].weights}')
    print(f'final model2 b0={model2.layers[0].biases}')
    print(f'final model2 w1={model2.layers[1].weights}')
    print(f'final model2 b1={model2.layers[1].biases}')
    print('\n')

    print('RUN SUCCESFULLY')

if __name__ == '__main__':

    Nclass = 500

    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X4 = np.random.randn(Nclass, 2) + np.array([0, 4])
    X = np.vstack([X1, X2, X3, X4])

    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass + [3] * Nclass)

    # let's see what it looks like
    #plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    #plt.show()

    # randomly initialize weights
    D = 2  # dimensionality of input
    M1 = 3  # hidden layer size
    M2 = 5  # hidden layer size
    K = 4  # number of classes
    W1 = np.random.randn(D, M1)
    b1 = np.random.randn(M1)
    W2 = np.random.randn(M1, K)
    b2 = np.random.randn(K)

    #print(f'generated w0={W1}')
    #print(f'generated b0={b1}')
    #print(f'generated w1={W2}')
    #print(f'generated b1={b2}')

    # turn Y into an indicator matrix for training
    N = len(Y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    #compare(AnnClassifier_2l_oryginal, AnnClassifier_2l_recursive, eps=1e-10, plot_training_history=False)


    l0 = DenseLayer(D, M1, SigmoidActivation(), init_params=[W1, b1])
    l1 = DenseLayer(M1, K, SoftmaxActivation(), init_params=[W2, b2])

    compare2(AnnClassifier_2l_oryginal, AnnClassifier_with_layers, [l0, l1], n_steps=100, eps=1e-10, plot_training_history=False)


    # randomly initialize weights

    W2_2 = np.random.randn(M1, M2)
    b2_2 = np.random.randn(M2)
    W3_2 = np.random.randn(M2, K)
    b3_2 = np.random.randn(K)

    #print(f'generated w0={W1}')
    #print(f'generated b0={b1}')
    #print(f'generated w1={W2}')
    #print(f'generated b1={b2}')



    l0 = DenseLayer(D, M1, SigmoidActivation(), init_params=[W1, b1])
    l1 = DenseLayer(M1, M2, SigmoidActivation(), init_params=[W2_2, b2_2])
    l2 = DenseLayer(M2, K, SoftmaxActivation(), init_params=[W3_2, b3_2])

    compare3(AnnClassifier_with_layers, [l0, l1, l2], n_steps=100, plot_training_history=True)

