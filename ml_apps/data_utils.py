import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from consts import TEST_DATA_PATH, RANDOM_STATE


def get_data_dir(file_name):
    return os.path.join(TEST_DATA_PATH, file_name)


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
