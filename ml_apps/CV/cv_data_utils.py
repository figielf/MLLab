import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_utils import get_data_dir, split_by_train_size


def plot_examples(x, y, cmap='gray', labels=None):
    plt.figure(figsize=(15, 15))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=cmap)
        if labels is None:
            plt.xlabel(y[i])
        else:
            plt.xlabel(labels[y[i]])
    plt.show()


def _get_mnist_data_raw():
    print("Reading in and transforming data...")
    df = pd.read_csv(get_data_dir('mnist.csv'))
    data = df.values
    X = data[:, 1:].astype(np.float32)
    Y = data[:, 0].astype(int)
    assert X.shape[1] == 28 * 28
    picture_shape = (28, 28)
    return X, Y, picture_shape


def get_mnist_data(train_size=0.8, should_plot_examples=True):
    assert train_size >= 0
    X, Y, picture_shape = _get_mnist_data_raw()
    X = np.divide(X, 255.0)  # data is from 0..255

    X_train, X_test, Y_train, Y_test = split_by_train_size(X, Y, train_size=train_size)
    if should_plot_examples:
        plot_examples(X.reshape((-1, *picture_shape)), Y, cmap='gray', labels=None)
    return X_train, X_test, Y_train, Y_test, picture_shape
