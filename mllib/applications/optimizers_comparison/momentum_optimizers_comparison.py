import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from activations import softmax, sigmoid
from integration_tests.utils.data_utils import get_mnist_normalized_data
from scores import accuracy, multiclass_cross_entropy
from utils_ndarray import ndarray_one_hot_encode, one_hot_2_vec


def _dW1(x, errors):
    return x.T.dot(errors)


def _db1(errors):
    return errors.sum(axis=0)


def _dW0(x, z, errors, W1):
    # return x.T.dot( ( errors.dot(W2.T) * ( z*(1 - z) ) ) ) # for sigmoid
    return x.T.dot( ( errors.dot(W1.T) * (z > 0) ) ) # for relu


def _db0(z, errors, W1):
    # return (errors.dot(W2.T) * ( z*(1 - z) )).sum(axis=0) # for sigmoid
    return (errors.dot(W1.T) * (z > 0)).sum(axis=0) # for relu


def _forward_log_odds(x, W, b):
    return x.dot(W) + b


class ANNfit():
    def __init__(self, W, b):
        self.W0 = W[0].copy()
        self.b0 = b[0].copy()
        self.W1 = W[1].copy()
        self.b1 = b[1].copy()

    def _forward(self, x):
        z1 = sigmoid(_forward_log_odds(x, self.W0, self.b0))
        return z1, softmax(_forward_log_odds(z1, self.W1, self.b1))

    def forward(self, x):
        _, y_hat = self._forward(x)
        return y_hat


class ANNfit_by_gd(ANNfit):
    def __init__(self, W, b):
        super().__init__(W, b)

    def fit_step(self, x, y, learning_rate, reg):
        batch_size = x.shape[0]
        z, p_hat = self._forward(x)
        errors = p_hat - y
        self.W1 = self.W1 - learning_rate * (_dW1(z, errors) / batch_size + reg * self.W1)
        self.b1 = self.b1 - learning_rate * (_db1(errors) / batch_size + reg * self.b1)
        self.W0 = self.W0 - learning_rate * (_dW0(x, z, errors, self.W1) / batch_size + reg * self.W0)
        self.b0 = self.b0 - learning_rate * (_db0(z, errors, self.W1) / batch_size + reg * self.b0)


class ANNfit_by_gd_with_momentum(ANNfit):
    def __init__(self, W, b):
        super().__init__(W, b)
        self.vW0 = np.zeros(self.W0.shape)
        self.vb0 = np.zeros(self.b0.shape)
        self.vW1 = np.zeros(self.W1.shape)
        self.vb1 = np.zeros(self.b1.shape)

    def fit_step(self, x, y, learning_rate, mu, reg):
        batch_size = x.shape[0]
        z, p_hat = self._forward(x)
        errors = p_hat - y

        dW1 = (_dW1(z, errors) / batch_size + reg * self.W1)
        db1 = (_db1(errors) / batch_size + reg * self.b1)
        dW0 = (_dW0(x, z, errors, self.W1) / batch_size + reg * self.W0)
        db0 = (_db0(z, errors, self.W1) / batch_size + reg * self.b0)

        self.vW0 = mu * self.vW0 - learning_rate * dW0
        self.vb0 = mu * self.vb0 - learning_rate * db0
        self.vW1 = mu * self.vW1 - learning_rate * dW1
        self.vb1 = mu * self.vb1 - learning_rate * db1

        self.W0 = self.W0 + self.vW0
        self.b0 = self.b0 + self.vb0
        self.W1 = self.W1 + self.vW1
        self.b1 = self.b1 + self.vb1


class ANNfit_by_gd_with_nesterov_momentum(ANNfit):
    def __init__(self, W, b):
        super().__init__(W, b)
        self.vW0 = np.zeros(self.W0.shape)
        self.vb0 = np.zeros(self.b0.shape)
        self.vW1 = np.zeros(self.W1.shape)
        self.vb1 = np.zeros(self.b1.shape)

    def fit_step(self, x, y, learning_rate, mu, reg):
        batch_size = x.shape[0]
        z, p_hat = self._forward(x)
        errors = p_hat - y

        dW1 = (_dW1(z, errors) / batch_size + reg * self.W1)
        db1 = (_db1(errors) / batch_size + reg * self.b1)
        dW0 = (_dW0(x, z, errors, self.W1) / batch_size + reg * self.W0)
        db0 = (_db0(z, errors, self.W1) / batch_size + reg * self.b0)

        self.vW0 = mu * self.vW0 - learning_rate * dW0
        self.vb0 = mu * self.vb0 - learning_rate * db0
        self.vW1 = mu * self.vW1 - learning_rate * dW1
        self.vb1 = mu * self.vb1 - learning_rate * db1

        self.W0 = self.W0 + self.vW0 - learning_rate * dW0
        self.b0 = self.b0 + self.vb0 - learning_rate * db0
        self.W1 = self.W1 + self.vW1 - learning_rate * dW1
        self.b1 = self.b1 + self.vb1 - learning_rate * db1


class ANNfit_by_adagrad(ANNfit):
    def __init__(self, W, b):
        super().__init__(W, b)
        self.cacheW0 = np.ones(self.W0.shape)
        self.cacheb0 = np.ones(self.b0.shape)
        self.cacheW1 = np.ones(self.W1.shape)
        self.cacheb1 = np.ones(self.b1.shape)
        self.eps = 1e-10

    def fit_step(self, x, y, learning_rate, reg):
        batch_size = x.shape[0]
        z, p_hat = self._forward(x)
        errors = p_hat - y

        dW1 = _dW1(z, errors) / batch_size + reg * self.W1
        db1 = _db1(errors) / batch_size + reg * self.b1
        dW0 = _dW0(x, z, errors, self.W1) / batch_size + reg * self.W0
        db0 = _db0(z, errors, self.W1) / batch_size + reg * self.b0

        self.cacheW1 = self.cacheW1 + dW1 * dW1
        self.cacheb1 = self.cacheb1 + db1 * db1
        self.cacheW0 = self.cacheW0 + dW0 * dW0
        self.cacheb0 = self.cacheb0 + db0 * db0

        self.W1 = self.W1 - learning_rate * dW1 / np.sqrt(self.cacheW1 + self.eps)
        self.b1 = self.b1 - learning_rate * db1 / np.sqrt(self.cacheb1 + self.eps)
        self.W0 = self.W0 - learning_rate * dW0 / np.sqrt(self.cacheW0 + self.eps)
        self.b0 = self.b0 - learning_rate * db0 / np.sqrt(self.cacheb0 + self.eps)


class ANNfit_by_RMSprop(ANNfit):
    def __init__(self, W, b):
        super().__init__(W, b)
        self.cacheW0 = np.ones(self.W0.shape)
        self.cacheb0 = np.ones(self.b0.shape)
        self.cacheW1 = np.ones(self.W1.shape)
        self.cacheb1 = np.ones(self.b1.shape)
        self.eps = 1e-10

    def fit_step(self, x, y, learning_rate, decay_rate, reg):
        batch_size = x.shape[0]
        z, p_hat = self._forward(x)
        errors = p_hat - y

        dW1 = _dW1(z, errors) / batch_size + reg * self.W1
        db1 = _db1(errors) / batch_size + reg * self.b1
        dW0 = _dW0(x, z, errors, self.W1) / batch_size + reg * self.W0
        db0 = _db0(z, errors, self.W1) / batch_size + reg * self.b0

        self.cacheW1 = decay_rate * self.cacheW1 + (1 - decay_rate) * dW1 * dW1
        self.cacheb1 = decay_rate * self.cacheb1 + (1 - decay_rate) * db1 * db1
        self.cacheW0 = decay_rate * self.cacheW0 + (1 - decay_rate) * dW0 * dW0
        self.cacheb0 = decay_rate * self.cacheb0 + (1 - decay_rate) * db0 * db0

        self.W1 = self.W1 - learning_rate * dW1 / np.sqrt(self.cacheW1 + self.eps)
        self.b1 = self.b1 - learning_rate * db1 / np.sqrt(self.cacheb1 + self.eps)
        self.W0 = self.W0 - learning_rate * dW0 / np.sqrt(self.cacheW0 + self.eps)
        self.b0 = self.b0 - learning_rate * db0 / np.sqrt(self.cacheb0 + self.eps)


def history_report(history, title=None, columns_to_report=['loss_train', 'loss_test', 'acc_train', 'acc_test']):
    print(f'\n{title}')
    if 'loss_train' in columns_to_report:
        print(f"Final train cost at {history['t'].values[-1]}: {history['loss_train'].values[-1]}")
    if 'loss_test' in columns_to_report:
        print(f"Final test cost at {history['t'].values[-1]}: {history['loss_test'].values[-1]}")
    if 'acc_train' in columns_to_report:
        print(f"Final train error rate at {history['t'].values[-1]}: {1 - history['acc_train'].values[-1]}")
    if 'acc_test' in columns_to_report:
        print(f"Final test error rate at {history['t'].values[-1]}: {1 - history['acc_test'].values[-1]}")


def report_histories(histories, titles, columns_to_report=['loss_train', 'loss_test', 'acc_train', 'acc_test'], columns_to_plot=['loss_train', 'loss_test', 'acc_train', 'acc_test']):
    assert len(histories) == len(titles)
    for history, title in zip(histories, titles):
        history_report(history, title=title, columns_to_report=columns_to_report)

    if columns_to_plot is not None:
        n_plots = len(columns_to_plot)
        figure_hight = int(np.ceil(n_plots / 2))
        plt.figure(figsize=(20, 20))
        i = 1
        for col in columns_to_plot:
            plt.subplot(figure_hight, 2, i)
            plt.title(col)
            for history, title in zip(histories, titles):
                if col in history.columns:
                    plt.plot(history['t'], history[col], label=title)
            plt.legend()
            i += 1
        plt.show()


def _calc_history(step, x, xtest, y, ytest, forward_model):
    p_hat = forward_model.forward(x)
    loss = multiclass_cross_entropy(y, p_hat)
    y_hat = np.argmax(p_hat, axis=1)
    p_hat_test = forward_model.forward(xtest)
    loss_test = multiclass_cross_entropy(ytest, p_hat_test)
    y_hat_test = np.argmax(p_hat_test, axis=1)
    acc = accuracy(one_hot_2_vec(y), y_hat)
    acc_test = accuracy(one_hot_2_vec(ytest), y_hat_test)
    return np.array([step, loss, loss_test, acc, acc_test])


def fit_gd(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, n_batches, learning_rate, reg,
                     calc_history_step=None, logging_step=None):
    print('Train parameters of ANN with minibatch gradient descent ...')
    print(f'fit_gd - W0.mean()={W0[0].mean()}, W0.std()={W0[0].std()}')
    print(f'fit_gd - b0.mean()={b0[0].mean()}, b0.std()={b0[0].std()}')
    print(f'fit_gd - W1.mean()={W0[1].mean()}, W0.std()={W0[1].std()}')
    print(f'fit_gd - b1.mean()={b0[1].mean()}, b0.std()={b0[1].std()}')
    n_samples = np.ceil(Xtrain.shape[0] / n_batches).astype(int)

    history = []
    iter_count = 0
    ann = ANNfit_by_gd(W0, b0)
    for epoch in range(n_epochs):
        Xtrain_shuffled, Ytrain_shuffled = shuffle(Xtrain, Ytrain)
        for batch in range(n_batches):
            if logging_step is not None and (iter_count % logging_step) == 0:
                print(f'----------epoch {epoch} - batch {batch}----------')
            idx_from = batch * n_samples
            idx_to = (batch + 1) * n_samples
            Xbatch, Ybatch = Xtrain_shuffled[idx_from:idx_to], Ytrain_shuffled[idx_from:idx_to]
            ann.fit_step(Xbatch, Ybatch, learning_rate, reg)

            if (iter_count % calc_history_step) == 0:
                history.append(_calc_history(iter_count, Xbatch, Xtest, Ybatch, Ytest, ann))
            iter_count += 1
    history = pd.DataFrame(history, columns=['t', 'loss_train', 'loss_test', 'acc_train', 'acc_test'])
    print(f'fit_minibatch_gd numer of steps: {iter_count}')
    return ann, history


def fit_gd_with_momentum(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, n_batches, learning_rate, mu, reg,
                     calc_history_step=None, logging_step=None):
    print('Train parameters of ANN with minibatch gradient descent with momentum ...')
    print(f'fit_gd - W0.mean()={W0[0].mean()}, W0.std()={W0[0].std()}')
    print(f'fit_gd - b0.mean()={b0[0].mean()}, b0.std()={b0[0].std()}')
    print(f'fit_gd - W1.mean()={W0[1].mean()}, W0.std()={W0[1].std()}')
    print(f'fit_gd - b1.mean()={b0[1].mean()}, b0.std()={b0[1].std()}')
    n_samples = np.ceil(Xtrain.shape[0] / n_batches).astype(int)

    history = []
    iter_count = 0
    ann = ANNfit_by_gd_with_momentum(W0, b0)
    for epoch in range(n_epochs):
        Xtrain_shuffled, Ytrain_shuffled = shuffle(Xtrain, Ytrain)
        for batch in range(n_batches):
            if logging_step is not None and (iter_count % logging_step) == 0:
                print(f'----------epoch {epoch} - batch {batch}----------')
            idx_from = batch * n_samples
            idx_to = (batch + 1) * n_samples
            Xbatch, Ybatch = Xtrain_shuffled[idx_from:idx_to], Ytrain_shuffled[idx_from:idx_to]
            ann.fit_step(Xbatch, Ybatch, learning_rate, mu, reg)

            if (iter_count % calc_history_step) == 0:
                history.append(_calc_history(iter_count, Xbatch, Xtest, Ybatch, Ytest, ann))
            iter_count += 1
    history = pd.DataFrame(history, columns=['t', 'loss_train', 'loss_test', 'acc_train', 'acc_test'])
    print(f'fit_minibatch_gd numer of steps: {iter_count}')
    return ann, history


def fit_gd_with_nesterov_momentum(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, n_batches, learning_rate, mu, reg,
                     calc_history_step=None, logging_step=None):
    print('Train parameters of ANN with minibatch gradient descent with nesterov momentum ...')
    print(f'fit_gd - W0.mean()={W0[0].mean()}, W0.std()={W0[0].std()}')
    print(f'fit_gd - b0.mean()={b0[0].mean()}, b0.std()={b0[0].std()}')
    print(f'fit_gd - W1.mean()={W0[1].mean()}, W0.std()={W0[1].std()}')
    print(f'fit_gd - b1.mean()={b0[1].mean()}, b0.std()={b0[1].std()}')
    n_samples = np.ceil(Xtrain.shape[0] / n_batches).astype(int)

    history = []
    iter_count = 0
    ann = ANNfit_by_gd_with_nesterov_momentum(W0, b0)
    for epoch in range(n_epochs):
        Xtrain_shuffled, Ytrain_shuffled = shuffle(Xtrain, Ytrain)
        for batch in range(n_batches):
            if logging_step is not None and (iter_count % logging_step) == 0:
                print(f'----------epoch {epoch} - batch {batch}----------')
            idx_from = batch * n_samples
            idx_to = (batch + 1) * n_samples
            Xbatch, Ybatch = Xtrain_shuffled[idx_from:idx_to], Ytrain_shuffled[idx_from:idx_to]
            ann.fit_step(Xbatch, Ybatch, learning_rate, mu, reg)

            if (iter_count % calc_history_step) == 0:
                history.append(_calc_history(iter_count, Xbatch, Xtest, Ybatch, Ytest, ann))
            iter_count += 1
    history = pd.DataFrame(history, columns=['t', 'loss_train', 'loss_test', 'acc_train', 'acc_test'])
    print(f'fit_minibatch_gd numer of steps: {iter_count}')
    return ann, history


def fit_adagrad(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, n_batches, learning_rate, reg,
                     calc_history_step=None, logging_step=None):
    print('Train parameters of ANN with minibatch gradient descent with nesterov momentum ...')
    print(f'fit_gd - W0.mean()={W0[0].mean()}, W0.std()={W0[0].std()}')
    print(f'fit_gd - b0.mean()={b0[0].mean()}, b0.std()={b0[0].std()}')
    print(f'fit_gd - W1.mean()={W0[1].mean()}, W0.std()={W0[1].std()}')
    print(f'fit_gd - b1.mean()={b0[1].mean()}, b0.std()={b0[1].std()}')
    n_samples = np.ceil(Xtrain.shape[0] / n_batches).astype(int)

    history = []
    iter_count = 0
    ann = ANNfit_by_adagrad(W0, b0)
    for epoch in range(n_epochs):
        Xtrain_shuffled, Ytrain_shuffled = shuffle(Xtrain, Ytrain)
        for batch in range(n_batches):
            if logging_step is not None and (iter_count % logging_step) == 0:
                print(f'----------epoch {epoch} - batch {batch}----------')
            idx_from = batch * n_samples
            idx_to = (batch + 1) * n_samples
            Xbatch, Ybatch = Xtrain_shuffled[idx_from:idx_to], Ytrain_shuffled[idx_from:idx_to]
            ann.fit_step(Xbatch, Ybatch, learning_rate, reg)

            if (iter_count % calc_history_step) == 0:
                history.append(_calc_history(iter_count, Xbatch, Xtest, Ybatch, Ytest, ann))
            iter_count += 1
    history = pd.DataFrame(history, columns=['t', 'loss_train', 'loss_test', 'acc_train', 'acc_test'])
    print(f'fit_minibatch_gd numer of steps: {iter_count}')
    return ann, history


def fit_rmsprop(Xtrain, Xtest, Ytrain, Ytest, W0, b0, n_epochs, n_batches, learning_rate, decay_rate, reg,
                     calc_history_step=None, logging_step=None):
    print('Train parameters of ANN with minibatch gradient descent with nesterov momentum ...')
    print(f'fit_gd - W0.mean()={W0[0].mean()}, W0.std()={W0[0].std()}')
    print(f'fit_gd - b0.mean()={b0[0].mean()}, b0.std()={b0[0].std()}')
    print(f'fit_gd - W1.mean()={W0[1].mean()}, W0.std()={W0[1].std()}')
    print(f'fit_gd - b1.mean()={b0[1].mean()}, b0.std()={b0[1].std()}')
    n_samples = np.ceil(Xtrain.shape[0] / n_batches).astype(int)

    history = []
    iter_count = 0
    ann = ANNfit_by_RMSprop(W0, b0)
    for epoch in range(n_epochs):
        Xtrain_shuffled, Ytrain_shuffled = shuffle(Xtrain, Ytrain)
        for batch in range(n_batches):
            if logging_step is not None and (iter_count % logging_step) == 0:
                print(f'----------epoch {epoch} - batch {batch}----------')
            idx_from = batch * n_samples
            idx_to = (batch + 1) * n_samples
            Xbatch, Ybatch = Xtrain_shuffled[idx_from:idx_to], Ytrain_shuffled[idx_from:idx_to]
            ann.fit_step(Xbatch, Ybatch, learning_rate, decay_rate, reg)

            if (iter_count % calc_history_step) == 0:
                history.append(_calc_history(iter_count, Xbatch, Xtest, Ybatch, Ytest, ann))
            iter_count += 1
    history = pd.DataFrame(history, columns=['t', 'loss_train', 'loss_test', 'acc_train', 'acc_test'])
    print(f'fit_minibatch_gd numer of steps: {iter_count}')
    return ann, history


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
    M = 300
    W0 = np.random.randn(D, M) / np.sqrt(D)
    b0 = np.zeros(M)
    W1 = np.random.randn(M, K) / np.sqrt(M)
    b1 = np.zeros(K)
    regularization = 0.01
    n_epochs = 50
    report_step = 5
    n_batches = 50

    histories = []
    titles = []

    mgd_model, mgd_fit_history = fit_gd(Xtrain, Xtest, Ytrain, Ytest, [W0, W1], [b0, b1],
                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                     learning_rate=0.001, reg=regularization,
                                                     calc_history_step=report_step, logging_step=report_step)
    histories.append(mgd_fit_history)
    titles.append('Minibatch gradient descent')

    mgdm_model, mgdm_fit_history = fit_gd_with_momentum(Xtrain, Xtest, Ytrain, Ytest, [W0, W1], [b0, b1],
                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                     learning_rate=0.001, mu=0.9, reg=regularization,
                                                     calc_history_step=report_step, logging_step=report_step)
    histories.append(mgdm_fit_history)
    titles.append('Minibatch gradient descent with momentum')

    mgdnm_model, mgdnm_fit_history = fit_gd_with_nesterov_momentum(Xtrain, Xtest, Ytrain, Ytest, [W0, W1], [b0, b1],
                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                     learning_rate=0.001, mu=0.9, reg=regularization,
                                                     calc_history_step=report_step, logging_step=report_step)
    histories.append(mgdnm_fit_history)
    titles.append('Minibatch gradient descent with nesterov momentum')

    ada_model, ada_fit_history = fit_adagrad(Xtrain, Xtest, Ytrain, Ytest, [W0, W1], [b0, b1],
                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                     learning_rate=0.001, reg=regularization,
                                                     calc_history_step=report_step, logging_step=report_step)
    histories.append(ada_fit_history)
    titles.append('AdaGrad')

    rms_model, rms_fit_history = fit_rmsprop(Xtrain, Xtest, Ytrain, Ytest, [W0, W1], [b0, b1],
                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                     learning_rate=0.001, decay_rate=0.999, reg=regularization,
                                                     calc_history_step=report_step, logging_step=report_step)
    histories.append(rms_fit_history)
    titles.append('RMSProp')

    cols = ['loss_train', 'loss_test', 'acc_train', 'acc_test']
    report_histories(histories, titles, columns_to_report=cols, columns_to_plot=cols)
