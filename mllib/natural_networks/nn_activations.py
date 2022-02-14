import numpy as np
from activations import sigmoid
from activations import softmax


class Activation:
    @staticmethod
    def value(x):
        pass

    @staticmethod
    def derivative(x):
        pass

    @staticmethod
    def backprop_derivative(x):
        pass


class SigmoidActivation(Activation):
    def __init__(self):
        super().__init__()

    @staticmethod
    def value(x):
        return sigmoid(x)

    @staticmethod
    def derivative(x):
        simg_x = sigmoid(x)
        return simg_x * (1 - simg_x)

    @staticmethod
    def backprop_derivative(x):
        return x * (1 - x)


class TanhActivation(Activation):
    def __init__(self):
        super().__init__()

    @staticmethod
    def value(x):
        ex = np.exp(x)
        e_x = np.exp(-x)
        return (ex - e_x) / (ex + e_x)

    @staticmethod
    def derivative(x):
        tanh_x = TanhActivation.value(x)
        return 1 - tanh_x * tanh_x

    @staticmethod
    def backprop_derivative(x):
        return 1 - x * x


class ReLUActivation(Activation):
    def __init__(self):
        super().__init__()

    @staticmethod
    def value(x):
        return x * ReLUActivation._step_fun(x)

    @staticmethod
    def derivative(x):
        return ReLUActivation._step_fun(x)

    @staticmethod
    def backprop_derivative(x):
        return ReLUActivation._step_fun(x)

    def _step_fun(x):
        result = np.zeros(x.shape)
        result[x > 0] = 1
        return result


class NoneActivation(Activation):
    def __init__(self):
        super().__init__()

    @staticmethod
    def value(x):
        return x

    @staticmethod
    def derivative(x):
        return x

    @staticmethod
    def backprop_derivative(x):
        return x


class SoftmaxActivation(Activation):
    def __init__(self):
        super().__init__()

    @staticmethod
    def value(x):
        return softmax(x, axis=1)

    @staticmethod
    def derivative(x):
        softmax_x = softmax(x, axis=1)
        return softmax_x * (1 - softmax_x)

    @staticmethod
    def backprop_derivative(x):
        raise NotImplemented()
