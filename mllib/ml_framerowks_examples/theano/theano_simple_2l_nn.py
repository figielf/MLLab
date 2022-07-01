import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tests.utils.data_utils import get_mnist_normalized_data
from utils_ndarray import ndarray_one_hot_encode


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def optimize_2l_nn(Xtrain, Xtest, Ytrain_ind, Ytest_ind, K, learning_rate, n_epochs=100, batch_size=500, reg=0.01):
    print_period = 10

    N, D = Xtrain.shape
    n_batches = N // batch_size

    M = 300
    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    # define theano variables and expressions
    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')

    # use the built-in theano functions to do relu and softmax
    thZ = relu( thX.dot(W1) + b1 ) # let's do it via numpy however relu is new in version 0.7.1
    thY = T.nnet.softmax( thZ.dot(W2) + b2 )

    # define the cost and prediction functions
    cost = -(thT * T.log(thY)).mean() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
    prediction = T.argmax(thY, axis=1)

    # define update parameters expressions using automatic differentiation (this includes also regularizatrion)
    update_W1 = W1 - learning_rate * T.grad(cost, W1)
    update_b1 = b1 - learning_rate * T.grad(cost, b1)
    update_W2 = W2 - learning_rate * T.grad(cost, W2)
    update_b2 = b2 - learning_rate * T.grad(cost, b2)

    # builds computation graph with 4 parameters to update by different update expressions
    train = theano.function(
        inputs=[thX, thT],
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)],
    )

    # create another function graph for this because we want it over the whole dataset
    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction],
    )

    costs = []
    for i in range(n_epochs):
        Xtrain_shuffled, Ytrain_ind_shuffled = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            Xbatch = Xtrain_shuffled[j*batch_size:(j*batch_size + batch_size),]
            Ybatch = Ytrain_ind_shuffled[j*batch_size:(j*batch_size + batch_size),]

            train(Xbatch, Ybatch)
            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print(f'Epoch:{i}, batch:{j}, cost={cost_val}, error rate={err}')
                costs.append(cost_val)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest, picture_shape = get_mnist_normalized_data(train_size=-1000, should_plot_examples=False)
    assert Xtrain.shape[1] == picture_shape[0] * picture_shape[1]

    K = 10
    Xtrain = Xtrain.astype(np.float32)
    Ytrain = Ytrain.astype(np.int32)
    Xtest = Xtest.astype(np.float32)
    Ytest = Ytest.astype(np.int32)
    Ytrain_ind = ndarray_one_hot_encode(Ytrain, K).astype(np.int32)
    Ytest_ind = ndarray_one_hot_encode(Ytest, K).astype(np.int32)

    optimize_2l_nn(Xtrain, Xtest, Ytrain_ind, Ytest_ind, K, learning_rate=0.01, n_epochs=100, batch_size=500, reg=0.01)
