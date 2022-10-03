import tensorflow as tf
import matplotlib.pyplot as plt

from natural_networks.dnn_estimator import dnn
from tests.utils.data_utils import get_mnist_data
from rbm import rbm

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

def print_history(Xtrain, Ytrain, Xtest, Ytest, _id, plt, pretrain):
    dnn_model = dnn(D, [1000, 750, 500], K, base_model=rbm)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        session.run(init_op)
        dnn_model.set_session(session)
        history = dnn_model.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=pretrain, n_epochs=3)

    plt.subplot(_n_runs, 2, 2 * _id + 1)
    plt.plot(history[:, 0], label='train cost')
    plt.plot(history[:, 1], label='test cost')
    plt.legend()
    plt.subplot(_n_runs, 2, 2 * _id + 2)
    plt.plot(history[:, 2], label='train error')
    plt.plot(history[:, 3], label='test error')
    plt.legend()
    plt.title(f'my dnn with my autoencoder - DNN training with pretrain={pretrain}')
    _id += 1


if __name__ == '__main__':
    K = 10
    test_size = 1000
    Xtrain, _, Ytrain, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain, Ytrain = Xtrain[:-test_size], Ytrain[:-test_size]
    Xtest, Ytest = Xtrain[-test_size:], Ytrain[-test_size:]

    _, D = Xtrain.shape
    K = len(set(Ytrain))

    _n_runs = 2
    plt.figure(figsize=(16, 16))
    print_history(Xtrain.copy(), Ytrain.copy(), Xtest.copy(), Ytest.copy(), 0, plt, pretrain=True)
    print_history(Xtrain.copy(), Ytrain.copy(), Xtest.copy(), Ytest.copy(), 1, plt, pretrain=False)
    plt.show()

