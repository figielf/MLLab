import itertools
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from consts import TEST_DATA_PATH, RANDOM_STATE, TEST_DATA_LARGE_FILES_PATH


def get_data_dir(file_name):
    return os.path.join(TEST_DATA_PATH, file_name)


def get_large_files_data_dir(file_name):
    return os.path.join(TEST_DATA_LARGE_FILES_PATH, file_name)


def split_by_train_size(X, Y, train_size, random_state=RANDOM_STATE, shuffle_data=True):
    if 0.0 < train_size < 1.0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - train_size, random_state=random_state)
    elif train_size == 1:
        X_train, Y_train = shuffle(X, Y, random_state=random_state)
        X_test, Y_test = None, None
    else:
        if isinstance(train_size, int):
            X, Y = shuffle(X, Y, random_state=random_state)
            X_train, Y_train = X[:train_size, :], Y[:train_size]
            X_test, Y_test = X[train_size:, :], Y[train_size:]
        else:
            raise Exception(f'Wrong test size value or type. Value:{train_size}, type:{type(train_size)}')

    if shuffle_data:
        if X_test is not None:
            X_train, X_test, Y_train, Y_test = shuffle(X_train, X_test, Y_train, Y_test, random_state=RANDOM_STATE)
        else:
            X_train, Y_train = shuffle(X_train, Y_train, random_state=RANDOM_STATE)
    return X_train, X_test, Y_train, Y_test


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          show_plot=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if show_plot:
        plt.show()
