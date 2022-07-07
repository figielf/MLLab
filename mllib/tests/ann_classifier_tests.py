import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from natural_networks.ann import ann_classifier
from tests.utils.data_utils import get_mnist_data

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


def unwind_history(history, n_layers):
    train_cost, test_cost, train_error, test_error, params = [], [], [], [], {}
    for k, v in history[0][4].items():
        params[k] = []

    for h in history:
        train_cost.append(h[0])
        test_cost.append(h[1])
        train_error.append(h[2])
        test_error.append(h[3])
        for k, v in h[4].items():
            params[k].append(v)
    return train_cost, test_cost, train_error, test_error, params

def calc_params_diff(params_history):
    W_diffs = []
    b_diffs = []
    for key, value in params_history.items():
        diffs = []
        old_val = value[0]
        for v in value[1:]:
            d = np.abs(v - old_val).mean()
            diffs.append(d)
            old_val = v
        if key.startswith('W'):
            W_diffs.append(diffs)
        else:
            b_diffs.append(diffs)
    return W_diffs, b_diffs

if __name__ == '__main__':
    K = 10
    test_size = 2000
    Xtrain, _, Ytrain, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain, Ytrain = Xtrain[:-test_size], Ytrain[:-test_size]
    Xtest, Ytest = Xtrain[-test_size:], Ytrain[-test_size:]

    _, D = Xtrain.shape
    K = len(set(Ytrain))
    #model = ann_classifier([1000, 750, 500])
    model = ann_classifier([256, 128, 64, 32, 16])
    with tf.compat.v1.Session() as session:
        model.set_session(session)
        history = model.fit(Xtrain.copy(), Ytrain.copy(), Xtest.copy(), Ytest.copy(), n_epochs=50, batch_size=1000, learning_rate=1e-4)

        train_cost, test_cost, train_error, test_error, params = unwind_history(history, n_layers=4)
        W_diffs, b_diffs = calc_params_diff(params)

        plt.figure(figsize=(16, 16))
        plt.subplot(2, 2, 1)
        plt.plot(train_cost, label='train cost')
        plt.plot(test_cost, label='test cost')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(train_error, label='train error')
        plt.plot(test_error, label='test error')
        plt.legend()
        plt.subplot(2, 2, 3)
        for i, v in enumerate(W_diffs):
            plt.plot(v, label=f'W param {i} layer diffs')
        plt.legend()
        plt.subplot(2, 2, 4)
        for i, v in enumerate(b_diffs):
            plt.plot(v, label=f'b param {i} layer diffs')
        plt.legend()
        plt.show()



