import numpy as np
from matplotlib import pyplot as plt
from activations import sigmoid, softmax
from scores import accuracy, log_loss
from utils_ndarray import ndarray_one_hot_encode


class AnnClassifier_with_layers:
    def __init__(self, layers, n_steps, n_classes=None, learning_rate=0.001,
                 plot_training_history=False):
        self.layers = layers
        self._n_steps = n_steps
        self._K = n_classes
        self._D = None
        self._learning_rate = learning_rate
        self._fit_history = None
        self._plot_training_history = plot_training_history

        self._target_label_size = None
        self.weights = None
        self.biases = None

    def fit(self, X, Y):
        assert isinstance(X, np.ndarray) and len(X.shape) == 2
        assert isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]

        if len(Y.shape) == 1:
            print('Assume targets is 1d array with values corresponding to K classes')
            Y_Kd = ndarray_one_hot_encode(Y, self._K)
            self._target_label_size = 1
        else:
            print('Assume targets is one hot encoded Kd array with columns corresponding to K classes')
            assert np.all(Y.sum(axis=1) == np.ones(Y.shape[0]))
            Y_Kd = Y
            self._target_label_size = Y_Kd.shape[1]

        self._D = X.shape[1]
        if self._K is None:
            self._K = len(set(Y))

        self._fit_history = self._backward(X, Y_Kd)

    def predict(self, X):
        assert isinstance(X, np.ndarray)
        p_hat = self._forward(X)
        y_hat_ind = np.argmax(p_hat, axis=1)
        if self._target_label_size == 1:
            return y_hat_ind
        else:
            return ndarray_one_hot_encode(y_hat_ind, self._target_label_size)

    def score(self, X, Y):
        y_hat = self.predict(X)
        return self._accuracy(Y, y_hat)

    def _accuracy(self, target, y_hat):
        assert isinstance(y_hat, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert target.shape == y_hat.shape
        assert len(target.shape) == 1 or len(target.shape) == 2
        if len(target.shape) == 1:
            return (y_hat == target).mean()
        return (y_hat == target).mean(axis=1)

    def _backward(self, x, y):
        hist = []
        # w0 = weights[0]
        # w1 = weights[1]
        # b0 = biases[0]
        # b1 = biases[1]
        print(f'-------------step 0-------------')
        z_layers = self._forward(x)  # , [w0, w1], [b0, b1], activations)
        p_hat = z_layers[-1]

        loss_start = log_loss(y, p_hat)
        acc_start = self._accuracy(np.argmax(y, axis=1), np.argmax(p_hat, axis=1))
        hist.append(np.array([loss_start, acc_start]))

        for step in range(1, self._n_steps + 1):
            print(f'-------------step {step}-------------')
            self._update_params(x, y, z_layers)

            z_layers = self._forward(x)  # , [w0, w1], [b0, b1], activations)
            p_hat = z_layers[-1]
            loss = log_loss(y, p_hat)
            acc = self._accuracy(np.argmax(y, axis=1), np.argmax(p_hat, axis=1))
            hist.append(np.array([loss, acc]))

        hist = np.array(hist)
        if self._plot_training_history:
            plt.figure(figsize=(10, 5))
            plt.plot(-1 * hist[:, 0], label='loss')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(hist[:, 1], label='accuracy')
            plt.legend()
            plt.show()
            print(f'final cost={hist[-1, 0]}, final accuracy={hist[-1, 1]}')

        return hist


    def _update_params(self, x, y, z_layers):
        n_layers = len(self.layers)

        deltas = []
        delta = y - z_layers[-1]
        deltas.append(delta)
        for l_idx in reversed(range(n_layers-1)):
            current_layer = self.layers[l_idx + 1]
            prev_layer = self.layers[l_idx]
            #delta = current_layer.calc_gradient_delta(next_layer_delta=delta, prev_layer=prev_layer, feature_values=z_layers[l_idx+1])
            delta = delta.dot(current_layer.weights.T) * prev_layer.activation.backprop_derivative(z_layers[l_idx+1])
            deltas.append(delta)
        deltas.reverse()

        for l_idx in range(n_layers):
            layer = self.layers[l_idx]
            layer.update_params(self._learning_rate, deltas[l_idx], z_layers[l_idx])


    def _forward(self, x):
        layers_out = []
        layers_out.append(x)
        z = x
        for layer in self.layers:
            z = layer.forward(z)
            layers_out.append(z)
        return layers_out

    #def _dw1(self, z, err, layer):
    #    dz = err
    #    return z.T.dot(dz)

    #def _db1(self, err, layer):
    #    return np.sum(err, axis=0)

    #def _dw0(self, x, z, err, w1, layer):
    #    dz = err.dot(w1.T) * layer.activation.backprop_derivative(z)
    #    return x.T.dot(dz)

    #def _db0(self, z, err, w1, layer):
    #    return np.sum(err.dot(w1.T) * layer.activation.backprop_derivative(z), axis=0)

    #def _dw(self, z, err, layer):
    #    return z.T.dot(err)

    #def _db(self, err, layer):
    #    return np.sum(err, axis=0)
