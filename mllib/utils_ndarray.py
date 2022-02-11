import numpy as np


def ndarray_one_hot_encode(vec, width):
    assert isinstance(vec, np.ndarray)
    assert len(vec.shape) == 1
    n = vec.shape[0]
    mat = np.zeros((n, width))
    mat[np.arange(n), vec] = 1
    return mat


def one_hot_2_vec(vec):
    assert isinstance(vec, np.ndarray)
    assert len(vec.shape) == 2
    return np.argmax(vec, axis=1)