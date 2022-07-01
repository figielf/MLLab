import numpy as np
import tensorflow as tf1
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tests.utils.data_utils import get_mnist_normalized_data
from utils_ndarray import ndarray_one_hot_encode


def error_rate(p, t):
    return np.mean(p != t)


def optimize_2l_nn(Xtrain, Xtest, Ytrain_ind, Ytest_ind, K, learning_rate, n_epochs=100, batch_size=500, reg=0.01):
    print_period = 10

    N, D = Xtrain.shape
    n_batches = N // batch_size

    M = 300
    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    # define variables and expressions
    X = tf1.placeholder(tf1.float32, shape=(None, D), name='X')
    T = tf1.placeholder(tf1.float32, shape=(None, K), name='T')
    W1 = tf1.Variable(W1_init.astype(np.float32))
    b1 = tf1.Variable(b1_init.astype(np.float32))
    W2 = tf1.Variable(W2_init.astype(np.float32))
    b2 = tf1.Variable(b2_init.astype(np.float32))

    # use the built-in theano functions to do relu and softmax
    Z = tf1.nn.relu( tf1.matmul(X, W1) + b1 )
    Y = tf1.matmul(Z, W2) + b2 # in tensoflow softmax is build into the cost function probably to omit exp calculation

    # softmax_cross_entropy_with_logits the 'logits', not softmax funstion output
    # if you wanted to know the actual output of the neural net, you can pass Y into tf.nn.softmax(logits)
    cost = tf1.reduce_sum(tf1.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=T))

    train_op = tf1.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9).minimize(cost)

    predict_op = tf1.argmax(Y, 1)

    costs = []
    init = tf1.global_variables_initializer()
    with tf1.Session() as session:
        session.run(init)

        for i in range(n_epochs):
            Xtrain_shuffled, Ytrain_ind_shuffled = shuffle(Xtrain, Ytrain_ind)
            for j in range(n_batches):
                Xbatch = Xtrain_shuffled[j*batch_size:(j*batch_size + batch_size),]
                Ybatch = Ytrain_ind_shuffled[j*batch_size:(j*batch_size + batch_size),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print(f'Epoch:{i}, batch:{j}, cost={test_cost}, error rate={err}')
                    costs.append(test_cost)

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

    optimize_2l_nn(Xtrain, Xtest, Ytrain_ind, Ytest_ind, K, learning_rate=0.001, n_epochs=100, batch_size=500, reg=0.01)

