import numpy as np

from utils_ndarray import one_hot_2_vec


def accuracy(target, y_hat):
    assert isinstance(y_hat, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert target.shape == y_hat.shape
    assert len(target.shape) == 1 or len(target.shape) == 2
    if len(target.shape) == 1:
        return (y_hat == target).mean()
    if len(target.shape) == 2:
        target_ind = one_hot_2_vec(target)
        y_hat_ind = one_hot_2_vec(y_hat)
        return (target_ind == y_hat_ind).mean()

def error_rate(target, y_hat):
    return 1 - accuracy(target, y_hat)

def binary_entropy(y):
    # assume y is binary - 0 or 1
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def exponential_loss(p_hat, y):
    return np.exp(-y * p_hat).mean()


def binary_cross_entropy(target, p_hat):
    return -(target * np.log(p_hat) + (1-target) * np.log(1-p_hat)).mean()


def multiclass_cross_entropy(target, p_hat):
    # target = [[0, 1, 0], [0, 0, 1]]
    # p_hat = [[0.04, 0.95, 0.01], [0.1, 0.8, 0.1]]
    # returns [-0.05129329, -2.30258509]
    assert isinstance(p_hat, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert target.shape == p_hat.shape
    assert len(target.shape) == 2 and target.shape[1] > 1
    log_p = np.log(p_hat[np.nonzero(target)])
    return -log_p.mean()


def r2(y, y_hat):
    assert isinstance(y, np.ndarray)
    assert isinstance(y_hat, np.ndarray)
    return 1 - sse(y, np.array([y.mean()])) / sse(y, y_hat)


def sse(y, y_hat):
    assert isinstance(y, np.ndarray)
    assert isinstance(y_hat, np.ndarray)
    assert len(y.shape) == 1 or y.shape[1] == 1
    err = (y - y_hat).flatten()
    result = err.dot(err)
    return result


def mse(y, y_hat):
    return sse(y, y_hat) / len(y)
