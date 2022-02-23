import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from activations import softmax
from integration_tests.utils.data_utils import get_mnist_normalized_data
from scores import accuracy, multiclass_cross_entropy
from utils_ndarray import ndarray_one_hot_encode, one_hot_2_vec


def _dW(x, errors):
    return x.T.dot(errors)


def _db(errors):
    return errors.sum(axis=0)


def _forward(X, W, b):
    return softmax(_forward_log_odds(X, W, b))


def _forward_log_odds(X, W, b):
    return X.dot(W) + b


def history_report(history, should_plot=True, title=None):
    print(title)
    print('Final train cost:', history['loss_train'].values[-1])
    print('Final test cost:', history['loss_test'].values[-1])
    print('Final train error rate:', 1 - history['acc_train'].values[-1])
    print('Final test error rate:', 1 - history['acc_test'].values[-1])

    if should_plot:
        plt.figure(figsize=(16, 16))
        create_history_figure(history, title)
        plt.show()


def create_history_figure(history, title=None, figure_hight=2, figure_width=1, figure_place_start=1):
    plt.subplot(figure_hight, figure_width, figure_place_start)
    plt.plot(history['loss_train'], label='train loss')
    plt.plot(history['loss_test'], label='test loss')
    if title is not None:
        plt.title(title + ' cost function per iteration')
    plt.legend()
    plt.subplot(figure_hight, figure_width, figure_place_start + 1)
    plt.plot(1 - history['acc_train'], label='train error rate')
    plt.plot(1 - history['acc_test'], label='test error rate')
    if title is not None:
        plt.title(title + ' error rate per iteration')
    plt.legend()


def plot_all_histories(histories, titles=None):
    n = len(histories)
    plt.figure(figsize=(20, 20))
    i=1
    for history, title in zip(histories, titles):
        create_history_figure(history, title, figure_hight=n, figure_width=2, figure_place_start=i)
        i += 2
    plt.show()

def _fit_step(x, y, weights, biases, learning_rate, reg):
    batch_size = x.shape[0]
    p_hat = _forward(x, weights, biases)
    errors = p_hat - y
    Wnext = weights - learning_rate * (_dW(x, errors) / batch_size + reg * weights)
    bnext = biases - learning_rate * (_db(errors) / batch_size + reg * biases)
    return Wnext, bnext


def _calc_history(x, xtest, y, ytest, weights, biases):
    p_hat = _forward(x, weights, biases)
    loss = multiclass_cross_entropy(y, p_hat)
    y_hat = np.argmax(p_hat, axis=1)
    p_hat_test = _forward(xtest, weights, biases)
    loss_test = multiclass_cross_entropy(ytest, p_hat_test)
    y_hat_test = np.argmax(p_hat_test, axis=1)
    acc = accuracy(one_hot_2_vec(y), y_hat)
    acc_test = accuracy(one_hot_2_vec(ytest), y_hat_test)
    return np.array([loss, loss_test, acc, acc_test])


def fit_gd(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, learning_rate, reg, logging_step=None):
    print('Train parameters of logistic Regression with gradient descent ...')
    print(f'simple_logistic_regression - W0.mean()={W0.mean()}, W0.std()={W0.std()}')
    print(f'simple_logistic_regression - b0.mean()={b0.mean()}, b0.std()={b0.std()}')
    W = W0.copy()
    b = b0.copy()
    history = []
    for epoch in range(n_epochs):
        if logging_step is not None and (epoch % logging_step) == 0:
            print(f'----------epoch {epoch}----------')
        W, b = _fit_step(Xtrain, Ytrain, W, b, learning_rate, reg)
        history.append(_calc_history(Xtrain, Xtest, Ytrain, Ytest, W, b))
    history = pd.DataFrame(history, columns=['loss_train', 'loss_test', 'acc_train', 'acc_test'])
    return W, b, history


def fit_minibatch_gd(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, n_batches, learning_rate, reg, logging_step=None):
    print('Train parameters of logistic Regression with minibatch gradient descent ...')
    print(f'simple_logistic_regression - W0.mean()={W0.mean()}, W0.std()={W0.std()}')
    print(f'simple_logistic_regression - b0.mean()={b0.mean()}, b0.std()={b0.std()}')
    W = W0.copy()
    b = b0.copy()
    n_samples = np.ceil(Xtrain.shape[0] / n_batches).astype(int)

    history = []
    for epoch in range(n_epochs):
        Xtrain_shuffled, Ytrain_shuffled = shuffle(Xtrain, Ytrain)
        i = 0
        for batch in  range(n_batches):
            if logging_step is not None and (i % logging_step) == 0:
                print(f'----------epoch {epoch} - batch {batch}----------')
            idx_from = batch * n_samples
            idx_to = (batch + 1) * n_samples
            Xbatch, Ybatch = Xtrain_shuffled[idx_from:idx_to], Ytrain_shuffled[idx_from:idx_to]
            W, b = _fit_step(Xbatch, Ybatch, W, b, learning_rate, reg)
            history.append(_calc_history(Xbatch, Xtest, Ybatch, Ytest, W, b))
            i += 1
    history = pd.DataFrame(history, columns=['loss_train', 'loss_test', 'acc_train', 'acc_test'])
    return W, b, history


def fit_sgd(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, learning_rate, reg, logging_step=None):
    print('Train parameters of logistic Regression with SGD (stochatic gradient descent) ...')
    print(f'simple_logistic_regression - W0.mean()={W0.mean()}, W0.std()={W0.std()}')
    print(f'simple_logistic_regression - b0.mean()={b0.mean()}, b0.std()={b0.std()}')
    W = W0.copy()
    b = b0.copy()
    n_samples = Xtrain.shape[0]

    history = []
    for epoch in range(n_epochs):
        Xtrain_shuffled, Ytrain_shuffled = shuffle(Xtrain, Ytrain)
        for sample in  range(n_samples):
            Xbatch, Ybatch = Xtrain_shuffled[sample].reshape(1,-1), Ytrain_shuffled[sample].reshape(1,-1)
            W, b = _fit_step(Xbatch, Ybatch, W, b, learning_rate, reg)
            if logging_step is not None and (sample % logging_step) == 0:
                print(f'----------epoch {epoch} - sample {sample} out of {n_samples}, progress {sample / n_samples * 100 : .2f}%----------')
                history.append(_calc_history(Xbatch, Xtest, Ybatch, Ytest, W, b))
    history = pd.DataFrame(history, columns=['loss_train', 'loss_test', 'acc_train', 'acc_test'])
    return W, b, history


def benchmark_full(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, learning_rate, reg, logging_step=None):
    print("Performing logistic regression...")
    print(f'simple_logistic_regression - W0.mean()={W0.mean()}, W0.std()={W0.std()}')
    print(f'simple_logistic_regression - b0.mean()={b0.mean()}, b0.std()={b0.std()}')
    W = W0.copy()
    b = b0.copy()

    history = []
    for epoch in range(n_epochs):
        if logging_step is not None and (epoch % logging_step) == 0:
            print(f'----------epoch {epoch}----------')
        p_y = _forward(Xtrain, W, b)
        n = len(Xtrain)
        W = W - learning_rate * (_dW(Xtrain, p_y - Ytrain) / n + reg * W)
        b = b - learning_rate * (_db(p_y - Ytrain) / n + reg * b)

        p_y = _forward(Xtrain, W, b)
        ll = multiclass_cross_entropy(Ytrain, p_y)

        p_y_test = _forward(Xtest, W, b)
        lltest = multiclass_cross_entropy(Ytest, p_y_test)

        acc = accuracy(p_y, Ytrain)
        acctest = accuracy(p_y_test, Ytest)

        history.append(np.array([ll, lltest, acc, acctest]))
    history = pd.DataFrame(history, columns=['loss_train', 'loss_test', 'acc_train', 'acc_test'])
    return W, b, history


if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest, picture_shape = get_mnist_normalized_data(train_size=-1000,
                                                                            should_plot_examples=False)
    assert Xtrain.shape[1] == picture_shape[0] * picture_shape[1]

    K = 10
    Ytrain = ndarray_one_hot_encode(Ytrain, K)
    Ytest = ndarray_one_hot_encode(Ytest, K)

    print('Xtrain.shape:', Xtrain.shape)
    print('Ytrain.shape:', Ytrain.shape)
    print('Xtest.shape:', Xtest.shape)
    print('Ytest.shape:', Ytest.shape)

    D = Xtrain.shape[1]
    W0 = np.random.randn(D, K) / np.sqrt(D)
    b0 = np.zeros(K)
    learning_rate = 0.0001
    regularization = 1.
    n_steps = 10
    n_batches = 10
    logging_step = int(n_steps/10)

    histories = []
    titles = []

    gd_bench_W, gd_bench_b, gd_bench_fit_history = benchmark_full(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs=n_steps, learning_rate=learning_rate, reg=regularization, logging_step=logging_step)
    print(f'final - W.mean()={gd_bench_W.mean()}, W0.std()={gd_bench_W.std()}')
    print(f'final - b.mean()={gd_bench_b.mean()}, b0.std()={gd_bench_b.std()}')
    histories.append(gd_bench_fit_history)
    titles.append('Vanila gradient descent - benchmark')

    gd_W, gd_b, gd_fit_history = fit_gd(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs=n_steps, learning_rate=learning_rate, reg=regularization, logging_step=logging_step)
    print(f'final - W.mean()={gd_W.mean()}, W0.std()={gd_W.std()}')
    print(f'final - b.mean()={gd_b.mean()}, b0.std()={gd_b.std()}')
    histories.append(gd_fit_history)
    titles.append('Vanila gradient descent')

    mgd_W, mgd_b, mgd_fit_history = fit_minibatch_gd(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs=int(n_steps/n_batches), n_batches=n_batches, learning_rate=learning_rate,
                                             reg=regularization, logging_step=logging_step)
    print(f'final - W.mean()={mgd_W.mean()}, W0.std()={mgd_W.std()}')
    print(f'final - b.mean()={mgd_b.mean()}, b0.std()={mgd_b.std()}')
    histories.append(mgd_fit_history)
    titles.append('CGD (stochatic gradient descent)')

    sgd_W, sgd_b, sgd_fit_history = fit_sgd(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs=1, learning_rate=learning_rate, reg=regularization, logging_step=int(len(Xtrain)/20))
    print(f'final - W.mean()={sgd_W.mean()}, W0.std()={sgd_W.std()}')
    print(f'final - b.mean()={sgd_b.mean()}, b0.std()={sgd_b.std()}')
    histories.append(sgd_fit_history)
    titles.append('CGD (stochatic gradient descent)')

    for history, title in zip(histories, titles):
        history_report(history, should_plot=True, title=title)

    plot_all_histories(histories, titles)