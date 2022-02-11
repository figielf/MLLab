import numpy as np
from natural_networks.nn_activations import Activation


class Layer:
    def __init__(self, imput_dim, output_dim, init_params):
        assert imput_dim is not None
        assert output_dim is not None
        self._input_dim = imput_dim
        self._output_dim = output_dim
        if init_params is not None:
            self._init_params = init_params
        else:
            self._init_params = self._initialize_params()
        self._params = self._init_params

    def forward(self, z):
        pass

    def d_params(self, z):
        pass

    def update_params(self, params):
        self._params = params

    def _initialize_params(self):
        pass

    def calc_gradient_delta(self):
        pass


class DenseLayer(Layer):
    def __init__(self, imput_dim, output_dim, activation, init_params=None):
        super().__init__(imput_dim, output_dim, init_params)
        if activation is not None:
            assert isinstance(activation, Activation)
        self.activation = activation
        assert len(self._params) == 2
        self.weights, self.biases = self._params


    def forward(self, x):
        assert x.shape[1] == self._input_dim
        a = x.dot(self.weights) + self.biases
        if self.activation is None:
            out = a
        else:
            out = self.activation.value(a)
        assert out.shape == (x.shape[0], self._output_dim)
        return out

    def d_params(self, err, prev_layer):
        dw = self._dw(z, err)
        db = self._db(err)
        pass

    def _dw(self, z, err):
        return z.T.dot(err)

    def _db(self, err):
        return np.sum(err, axis=0)

    def _initialize_params(self):
        w0 = np.random.randn(self._imput_dim, self._output_dim) / np.sqrt(self._output_dim)
        b0 = np.random.randn(self._output_dim)
        return w0, b0

    def update_params(self, learning_rate, delta, z):
        d_w = z.T.dot(delta)
        d_b = np.sum(delta, axis=0)
        self.weights = self.weights + learning_rate * d_w
        self.biases = self.biases + learning_rate * d_b
        super().update_params((self.weights, self.biases))

    def calc_gradient_delta(self, next_layer_delta, prev_layer, feature_values):
        return next_layer_delta.dot(self.weights.T) * prev_layer.activation.backprop_derivative(feature_values)
