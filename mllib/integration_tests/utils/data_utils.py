import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from future.utils import iteritems
from integration_tests.consts import TEST_DATA_PATH


def get_data_dir(file_name):
    return os.path.join(TEST_DATA_PATH, file_name)


def get_housing_data(test_size=0.3):
    class HousingDataTransformer:
        def __init__(self, numerical_columns):
            self.numerical_columns = numerical_columns
            self.transformers = None
            self.features_dim = None

        def fit(self, X):
            self.transformers = []
            self.features_dim = X.shape[1]
            for c in range(self.features_dim):
                if X.columns[c] in self.numerical_columns:
                    scaler = StandardScaler()
                    scaler.fit(X.iloc[:, c].values.reshape(-1, 1))
                    self.transformers.append(scaler)
                else:
                    self.transformers.append(None)

        def transform(self, X):
            result = np.zeros((len(X), self.features_dim))
            i = 0
            for c in range(self.features_dim):
                scaler = self.transformers[c]
                if X.columns[c] in self.numerical_columns:
                    result[:, i] = scaler.transform(X.iloc[:, c].values.reshape(-1, 1)).flatten()
                else:
                    result[:, i] = X.iloc[:, c]
                i += 1
            return result

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    housing_df = pd.read_csv(get_data_dir('housing.data'), header=None, delim_whitespace=True)
    housing_df.columns = [
        'crim',  # numerical
        'zn',  # numerical
        'nonretail',  # numerical
        'river',  # binary
        'nox',  # numerical
        'rooms',  # numerical
        'age',  # numerical
        'dis',  # numerical
        'rad',  # numerical
        'tax',  # numerical
        'ptratio',  # numerical
        'b',  # numerical
        'lstat',  # numerical
        'medv',  # numerical -- this is the target
    ]

    if housing_df.isna().sum().max() == 0:
        print('There is no NA values')
    else:
        print(f'There are {housing_df.isna().sum().max()} NAs')

    # NO_TRANSFORM = ['river']

    HOUSING_NUMERICAL_COLS = [
        'crim',  # numerical
        'zn',  # numerical
        'nonretail',  # numerical
        'nox',  # numerical
        'rooms',  # numerical
        'age',  # numerical
        'dis',  # numerical
        'rad',  # numerical
        'tax',  # numerical
        'ptratio',  # numerical
        'b',  # numerical
        'lstat',  # numerical
    ]

    df = housing_df.copy()
    X = df.iloc[:, :-1]
    Y = df['medv']
    transformer = HousingDataTransformer(HOUSING_NUMERICAL_COLS)

    if test_size > 0.0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, Y_train, Y_test = X, None, Y, None

    X_train_transformed = transformer.fit_transform(X_train)
    if X_test is not None:
        return X_train_transformed, transformer.transform(X_test), Y_train.values, Y_test.values
    else:
        return X_train_transformed, None, Y_train.values, None


def get_mushroom_data(test_size=0.3):
    def replace_missing(df, numerical_columns, categorical_columns, special_missing_category='missing'):
        # standard method of replacement for numerical columns is median
        for col in numerical_columns:
            if np.any(df[col].isnull()):
                med = np.median(df[col][df[col].notnull()])
                df.loc[df[col].isnull(), col] = med

        # set a special value = 'missing'
        for col in categorical_columns:
            if np.any(df[col].isnull()):
                print(col)
                df.loc[df[col].isnull(), col] = special_missing_category

    class MushroomDataTransformer:
        def __init__(self, numerical_columns, categorical_columns):
            self.numerical_columns = numerical_columns
            self.categorical_columns = categorical_columns
            self.labelEncoders = None
            self.scalers = None
            self.D = None

        def fit(self, df):
            self.labelEncoders = {}
            self.scalers = {}
            for col in self.numerical_columns:
                scaler = StandardScaler()
                scaler.fit(df[col].reshape(-1, 1))
                self.scalers[col] = scaler

            for col in self.categorical_columns:
                encoder = LabelEncoder()
                # in case the train set does not have 'missing' value but test set does
                values = df[col].tolist()
                values.append('missing')
                encoder.fit(values)
                self.labelEncoders[col] = encoder

            # find dimensionality
            self.D = len(self.numerical_columns)
            for col, encoder in iteritems(self.labelEncoders):
                self.D += len(encoder.classes_)
            print("dimensionality:", self.D)

        def transform(self, df):
            N, _ = df.shape
            X = np.zeros((N, self.D))
            i = 0
            for col, scaler in iteritems(self.scalers):
                X[:, i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
                i += 1

            for col, encoder in iteritems(self.labelEncoders):
                K = len(encoder.classes_)
                X[np.arange(N), encoder.transform(df[col]) + i] = 1
                i += K
            return X

        def fit_transform(self, df):
            self.fit(df)
            return self.transform(df)

    mushroom_df = pd.read_csv(get_data_dir('agaricus-lepiota.data'), header=None)

    if mushroom_df.isna().sum().max() == 0:
        print('There is no NA values')
    else:
        print(f'There are {mushroom_df.isna().sum().max()} NAs')

    MUSHROOM_NUMERICAL_COLS = ()
    MUSHROOM_CATEGORICAL_COLS = np.arange(22) + 1  # 1..22 inclusive

    df = mushroom_df.copy()
    # replace label column: e/p --> 0/1, e = edible = 0, p = poisonous = 1
    df[0] = df.apply(lambda row: 0 if row[0] == 'e' else 1, axis=1)

    replace_missing(df, MUSHROOM_NUMERICAL_COLS, MUSHROOM_CATEGORICAL_COLS)
    transformer = MushroomDataTransformer(MUSHROOM_NUMERICAL_COLS, MUSHROOM_CATEGORICAL_COLS)

    X = df
    Y = df[0]
    if test_size > 0.0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, Y_train, Y_test = X, None, Y, None

    X_train_transformed = transformer.fit_transform(X_train)
    if X_test is not None:
        return X_train_transformed, transformer.transform(X_test), Y_train.values, Y_test.values
    else:
        return X_train_transformed, None, Y_train.values, None


def get_mnist_data(should_shuffle=True, should_plot_examples=True):
    print("Reading in and transforming data...")
    df = pd.read_csv(get_data_dir('mnist.csv'))
    data = df.values
    if should_shuffle:
        np.random.shuffle(data)
    X = np.divide(data[:, 1:], 255.0)  # data is from 0..255
    Y = data[:, 0]
    assert X.shape[1] == 28 * 28
    picture_shape = (28, 28)

    if should_plot_examples:
        plot_examples(X.reshape((-1, *picture_shape)), Y, cmap='gray', labels=None)
    return X, Y, picture_shape


def get_xor_data(N=200, should_plot_data=True):
    X = np.zeros((N, 2))
    Nq = N // 4
    X[:Nq] = np.random.random((Nq, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[Nq:2 * Nq] = np.random.random((Nq, 2)) / 2  # (0-0.5, 0-0.5)
    X[2 * Nq:3 * Nq] = np.random.random((Nq, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[3 * Nq:] = np.random.random((Nq, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    Y = np.array([0] * (N // 2) + [1] * (N // 2))

    X, Y = shuffle_pairs(X, Y)

    if should_plot_data:
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
        plt.title('Training data plot')
        plt.show()
    return X, Y


def get_simple_xor_data(should_plot_data=True):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    if should_plot_data:
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
        plt.title('Training data plot')
        plt.show()
    return X, Y


def get_donut_data(N=200, should_plot_data=True):
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))

    X, Y = shuffle_pairs(X, Y)

    if should_plot_data:
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
        plt.title('Training data plot')
        plt.show()
    return X, Y


def shuffle_pairs(X, Y):
    N = len(X)
    indexes = np.arange(N)
    np.random.shuffle(indexes)
    X = X[indexes]
    Y = Y[indexes]
    return X, Y


def plot_misclasified_examples(x, true_lables, predicted_lables, n=5, print_misclassified=False, labels=None):
    misclassified_idx = np.where(predicted_lables != true_lables)[0]
    misclassified_random_idxes = np.random.choice(misclassified_idx, n * n)
    plt.figure(figsize=(15, 15))
    for i in range(n * n):
        idx = misclassified_random_idxes[i]
        plt.subplot(n, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[idx], cmap='gray')
        if labels is None:
            plt.xlabel("True  %s, Pred: %s" % (true_lables[idx], predicted_lables[idx]))
        else:
            plt.xlabel("True  %s, Pred: %s" % (labels[true_lables[idx]], labels[predicted_lables[idx]]))
    plt.show()

    if print_misclassified:
        if labels is None:
            print(pd.DataFrame({'idx': misclassified_random_idxes,
                                'true': true_lables[misclassified_random_idxes],
                                'pred': predicted_lables[misclassified_random_idxes]}))
        else:
            print(pd.DataFrame({'idx': misclassified_random_idxes,
                                'true': true_lables[misclassified_random_idxes],
                                'pred': predicted_lables[misclassified_random_idxes]}))


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
