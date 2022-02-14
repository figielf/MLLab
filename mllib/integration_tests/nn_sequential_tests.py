import numpy as np
from matplotlib import pyplot as plt

from natural_networks.nn_activations import SigmoidActivation, SoftmaxActivation
from natural_networks.nn_layers import DenseLayer
from natural_networks.nn_sequential_estimator import NNSequentialClassifier
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

    # randomly initialize weights
    D = 2  # dimensionality of input
    M1 = 3  # hidden layer size
    M2 = 5  # hidden layer size
    K = 4  # number of classes

    Ttrain = ndarray_one_hot_encode(Ytrain, K)
    Ttest = ndarray_one_hot_encode(Ytest, K)

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

    model2l = NNSequentialClassifier(layers=[l0, l1], n_steps=100, n_classes=K, plot_training_history=True)

    model2l.fit(Xtrain, Ytrain)
    assert np.all(model2l.score(Xtrain, model2l.predict(Xtrain))) == 1

    print('\n')
    print(f'final 2 layers model w0={model2l.layers[0].weights}')
    print(f'final 2 layers model b0={model2l.layers[0].biases}')
    print(f'final 2 layers model w1={model2l.layers[1].weights}')
    print(f'final 2 layers model b1={model2l.layers[1].biases}')

    model2l_2 = NNSequentialClassifier(layers=[l0, l1], n_steps=100, n_classes=K, plot_training_history=False)
    model2l_2.fit(Xtrain, Ttrain)


    print('\n')
    print('Train accuracy in 2 layers nn:', model2l.score(Xtrain, Ytrain))
    print('Test accuracy in 2 layers nn:', model2l.score(Xtest, Ytest))
    print('Train accuracy 2 in 2 layers nn:', model2l_2.score2(Xtrain, Ttrain))
    print('Test accuracy 2 in 2 layers nn:', model2l_2.score2(Xtest, Ttest))


    ## 3 layers NN
    l0 = DenseLayer(D, M1, SigmoidActivation())#, init_params=[W1, b1])
    l1 = DenseLayer(M1, M2, SigmoidActivation())#, init_params=[W2_2, b2_2])
    l2 = DenseLayer(M2, K, SoftmaxActivation())#, init_params=[W3_2, b3_2])
    model3l = NNSequentialClassifier(layers=[l0, l1, l2], n_steps=100, n_classes=K, plot_training_history=True)

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
