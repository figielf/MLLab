import numpy as np
from natural_networks.nn_activations import Activation


class Layer:
    def __init__(self, imput_dim, output_dim):
        assert imput_dim is not None
        assert output_dim is not None
        self._input_dim = imput_dim
        self._output_dim = output_dim

    def forward(self, z):
        pass

    def initialize_params(self):
        pass

    def calc_gradient_delta(self, next_layer_delta, prev_layer, feature_values):
        pass

    def update_params(self, learning_rate, delta, z):
        pass


class DenseLayer(Layer):
    def __init__(self, imput_dim, output_dim, activation, init_params=None):
        super().__init__(imput_dim, output_dim)
        if activation is not None:
            assert isinstance(activation, Activation)
        self.activation = activation
        if init_params is not None:
            assert len(init_params) == 2
            self.weights, self.biases = init_params
        else:
            self.initialize_params()

    def forward(self, x):
        assert x.shape[1] == self._input_dim
        a = x.dot(self.weights) + self.biases
        if self.activation is None:
            out = a
        else:
            out = self.activation.value(a)
        if self._output_dim == 1:
            assert out.shape[0] == x.shape[0]
            out = out.reshape(-1, self._output_dim)
        else:
            assert out.shape == (x.shape[0], self._output_dim)
        return out

    def _dw(self, z, err):
        return z.T.dot(err)

    def _db(self, err):
        return np.sum(err, axis=0)

    def initialize_params(self):
        self.weights = np.random.randn(self._input_dim, self._output_dim) / np.sqrt(self._output_dim)
        self.biases = np.random.randn(self._output_dim)

    def update_params(self, learning_rate, delta, z):
        d_w = z.T.dot(delta)
        d_b = np.sum(delta, axis=0)
        self.weights = self.weights + learning_rate * d_w
        self.biases = self.biases + learning_rate * d_b

    def calc_gradient_delta(self, next_layer_delta, prev_layer, feature_values):
        return next_layer_delta.dot(self.weights.T) * prev_layer.activation.backprop_derivative(feature_values)
