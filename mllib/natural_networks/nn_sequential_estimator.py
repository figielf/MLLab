import numpy as np
from matplotlib import pyplot as plt
from scores import multiclass_cross_entropy, mse, r2
from utils_ndarray import ndarray_one_hot_encode, one_hot_2_vec


class NNSequential:
    def __init__(self, layers, n_steps, learning_rate=0.001, plot_training_history=False):
        self.layers = layers
        self._n_steps = n_steps
        self._learning_rate = learning_rate
        self._fit_history = None
        self._plot_training_history = plot_training_history
        self.fit_history = None

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

    def score(self, X, Y):
        pass

    def _fit_prepare(self, X, Y):
        assert isinstance(X, np.ndarray) and len(X.shape) == 2
        assert isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]
        self.fit_history = []

    def _backward(self, x, y):
        # print(f'-------------step 0-------------')
        feature_values_by_layer = self._forward(x)
        self._calc_fit_scores(y, feature_values_by_layer[-1])

        for step in range(1, self._n_steps + 1):
            # print(f'-------------step {step}-------------')
            self._update_params(y, feature_values_by_layer)

            feature_values_by_layer = self._forward(x)
            self._calc_fit_scores(y, feature_values_by_layer[-1])

        hist = np.array(self.fit_history)
        if self._plot_training_history:
            self._plot_history(hist)

        return hist

    def _plot_history(self, history):
        pass

    def _calc_fit_scores(self, forward_result, p_hat):
        pass

    def _update_params(self, y, z_by_layers):
        n_layers = len(self.layers)

        deltas = []
        deltas.append(y - z_by_layers[-1])
        for l_idx in reversed(range(n_layers - 1)):
            current_layer = self.layers[l_idx]
            current_layer_z = z_by_layers[l_idx + 1]
            following_layer = self.layers[l_idx + 1]
            following_layer_delta = deltas[-1]

            back_propagated_delta = following_layer.propagate_delta_back(following_layer_delta)
            deltas.append(current_layer.calc_gradient_delta(back_propagated_delta,  current_layer_z))
        deltas.reverse()

        for l_idx in range(n_layers):
            layer = self.layers[l_idx]
            layer.update_params(self._learning_rate, deltas[l_idx], z_by_layers[l_idx])

    def _forward(self, x):
        out_feature_values_by_layer = [x]
        z = x
        for layer in self.layers:
            z = layer.forward(z)
            out_feature_values_by_layer.append(z)
        return out_feature_values_by_layer


class NNSequentialClassifier(NNSequential):
    def __init__(self, layers, n_steps, n_classes=None, learning_rate=0.001,
                 plot_training_history=False):
        super().__init__(layers, n_steps, learning_rate, plot_training_history)
        self._K = n_classes
        self._target_label_size = None

    def fit(self, X, Y):
        super()._fit_prepare(X, Y)

        if len(Y.shape) == 1:
            print('Assume targets is 1d array with values corresponding to K classes')
            Y_Kd = ndarray_one_hot_encode(Y, self._K)
            self._target_label_size = 1
        else:
            print('Assume targets is one hot encoded Kd array with columns corresponding to K classes')
            assert np.all(Y.sum(axis=1) == np.ones(Y.shape[0]))
            Y_Kd = Y
            self._target_label_size = Y_Kd.shape[1]

        if self._K is None:
            self._K = len(set(Y))

        return self._backward(X, Y_Kd)

    def predict(self, X):
        assert isinstance(X, np.ndarray)
        y_hat_ind = self._predict_ind(X)
        if self._target_label_size == 1:
            return y_hat_ind
        else:
            return ndarray_one_hot_encode(y_hat_ind, self._target_label_size)

    def _predict_ind(self, X):
        layers_output = self._forward(X)
        p_hat = layers_output[-1]
        return np.argmax(p_hat, axis=1)

    def score(self, X, Y):
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]
        if self._target_label_size == 1:
            assert len(Y.shape) == 1
            T_ind = Y
        else:
            assert len(Y.shape) == 2 and Y.shape[1] > 1
            T_ind = one_hot_2_vec(Y)
        y_hat_ind = self._predict_ind(X)
        return self._accuracy_ind(T_ind, y_hat_ind)

    @staticmethod
    def _accuracy_ind(target, y_hat_ind):
        return (target == y_hat_ind).mean()

    def _calc_fit_scores(self, y, p_hat):
        loss = multiclass_cross_entropy(y, p_hat)
        acc = self._accuracy_ind(np.argmax(y, axis=1), np.argmax(p_hat, axis=1))
        self.fit_history.append(np.array([loss, acc]))

    def _plot_history(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(-1 * history[:, 0], label='loss')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(history[:, 1], label='accuracy')
        plt.legend()
        plt.show()
        print(f'final cost={history[-1, 0]}, final accuracy={history[-1, 1]}')


class NNSequentialRegressor(NNSequential):
    def __init__(self, layers, n_steps, learning_rate=0.001, plot_training_history=False):
        super().__init__(layers, n_steps, learning_rate, plot_training_history)

    def fit(self, X, Y):
        super()._fit_prepare(X, Y)
        assert len(Y.shape) == 1 or (len(Y.shape) == 2 and Y.shape[1] == 1)
        self._initial_target_shape = Y.shape
        return self._backward(X, Y.reshape(-1, 1))

    def predict(self, X):
        assert isinstance(X, np.ndarray)
        layers_out = self._forward(X)
        y_hat = layers_out[-1]
        assert len(y_hat.shape) == 2 and y_hat.shape[1] == 1
        if len(self._initial_target_shape) == 1:
            return y_hat.flatten()
        else:
            return y_hat.reshape(-1, 1)

    def score(self, X, Y):
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]
        y_hat = self.predict(X)
        return mse(Y, y_hat)

    def _calc_fit_scores(self, y, y_hat):
        y_hat = y_hat.reshape(y.shape)
        loss = mse(y, y_hat)
        R2 = r2(y, y_hat)
        self.fit_history.append(np.array([loss, R2]))

    def _plot_history(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(-1 * history[:, 0], label='mse')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(history[:, 1], label='R2')
        plt.legend()
        plt.show()
        print(f'final cost MSE={history[-1, 0]}, final R2={history[-1, 1]}')
