import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=1):
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis=axis, keepdims=True)
