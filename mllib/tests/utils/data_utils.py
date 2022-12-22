import os
import string
import wave

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from tests.consts import TEST_DATA_PATH

RANDOM_STATE = 123


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

    if 0.0 < train_size < 1.0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - train_size, random_state=RANDOM_STATE)
    elif train_size == 1:
        X_train, Y_train = shuffle(X, Y, random_state=RANDOM_STATE)
        X_test, Y_test = None, None
    else:
        if isinstance(train_size, int):
            X, Y = shuffle(X, Y, random_state=RANDOM_STATE)
            X_train, Y_train = X[:train_size, :], Y[:train_size]
            X_test, Y_test = X[train_size:, :], Y[train_size:]
        else:
            raise Exception(f'Wrong test size value or type. Value:{train_size}, type:{type(train_size)}')

    if should_plot_examples:
        plot_examples(X.reshape((-1, *picture_shape)), Y, cmap='gray', labels=None)
    return X_train, X_test, Y_train, Y_test, picture_shape


def get_mnist_normalized_data(train_size=0.8, should_plot_examples=True):
    assert train_size != 0
    X, Y, picture_shape = _get_mnist_data_raw()

    if 0.0 < train_size < 1.0:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 - train_size, random_state=RANDOM_STATE)
    elif train_size == 1:
        X_train, Y_train = shuffle(X, Y, random_state=RANDOM_STATE)
        X_test, Y_test = None, None
    else:
        if isinstance(train_size, int):
            X, Y = shuffle(X, Y, random_state=RANDOM_STATE)
            X_train, Y_train = X[:train_size, :], Y[:train_size]
            X_test, Y_test = X[train_size:, :], Y[train_size:]
        else:
            raise Exception(f'Wrong test size value or type. Value:{train_size}, type:{type(train_size)}')

    # normalize the data
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    np.place(std, std == 0, 1)
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std

    if should_plot_examples:
        plot_examples(X.reshape((-1, *picture_shape)), Y, cmap='gray', labels=None)
    return X_train, X_test, Y_train, Y_test, picture_shape


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


def get_cloud_3d_data(samples_per_cloud=100, should_plot_data=True):
    # define the centers of each Gaussian cloud
    centers = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]) * 3

    # create the clouds, Gaussian samples centered at
    # each of the centers we just made
    X = []
    for c in centers:
        cloud = np.random.randn(samples_per_cloud, 3) + c
        X.append(cloud)
    X = np.concatenate(X)

    # visualize the clouds in 3-D
    # add colors / labels so we can track where the points go
    Y = np.array([[i] * samples_per_cloud for i in range(len(centers))]).flatten()
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


def get_book_titles_data():
    titles = []
    stop_words = []

    with open(get_data_dir('all_book_titles.txt')) as file:
        for line in file:
            title = line.rstrip()
            title = title.encode('ascii', 'ignore')  # this will throw exception if bad characters
            title = title.decode('utf-8')
            titles.append(title)

    with open(get_data_dir('stopwords.txt')) as file:
        for line in file:
            sw = line.rstrip()
            sw = sw.encode('ascii', 'ignore')
            sw = sw.decode('utf-8')
            stop_words.append(sw)

    return titles, set(stop_words)


def get_edgar_allan_poe_data():
    poem_lines = []
    with open(get_data_dir('edgar_allan_poe.txt')) as file:
        for line in file:
            txt = line.rstrip().lower()
            if txt:
                txt = txt.translate(str.maketrans('', '', string.punctuation))
                poem_lines.append(txt)
    return poem_lines


def get_robert_frost_data():
    poem_lines = []
    with open(get_data_dir('robert_frost.txt')) as file:
        for line in file:
            txt = line.rstrip().lower()
            if txt:
                txt = txt.translate(str.maketrans('', '', string.punctuation))
                poem_lines.append(txt)
    return poem_lines


def get_edgar_allan_and_robert_frost_data(test_size=0.2):
    poem_lines = []
    for txt in get_edgar_allan_poe_data():
        poem_lines.append((txt, 0))
    for txt in get_robert_frost_data():
        poem_lines.append((txt, 1))

    txt_df = pd.DataFrame(poem_lines, columns=['txt', 'author'])
    if test_size > 0.0:
        X_train, X_test, Y_train, Y_test = train_test_split(txt_df['txt'].values, txt_df['author'].values,
                                                            test_size=test_size, random_state=RANDOM_STATE)
    else:
        X_train, X_test, Y_train, Y_test = txt_df['txt'].values, None, txt_df['author'].values, None

    return X_train, X_test, Y_train, Y_test


def get_coin_flip_data():
    X = []
    for line in open(get_data_dir('coin_data.txt')):
        # 1 for H, 0 for T
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)
    return X


def get_helloworld_data():
    spf = wave.open(get_data_dir('helloworld.wav'), 'r')
    # Extract Raw Audio from Wav File
    # If you right-click on the file and go to "Get Info", you can see:
    # sampling rate = 16000 Hz
    # bits per sample = 16
    # The first is quantization in time
    # The second is quantization in amplitude
    # We also do this for images!
    # 2^16 = 65536 is how many different sound levels we have
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'int16')
    return signal


def get_titanic_data_raw(train_size=0.8):
    assert train_size >= 0
    titanic_train_df = pd.read_csv(get_data_dir('titanic_train.csv'))
    #titanic_test_df = pd.read_csv(get_data_dir('titanic_test.csv'))
    X = titanic_train_df.drop(columns=['Survived'])
    Y = titanic_train_df['Survived']
    #X_test = titanic_test_df.drop(columns=['Survived'])
    #Y_test = titanic_test_df['Survived']

    X_train, X_test, Y_train, Y_test = split_by_train_size(X, Y, train_size, RANDOM_STATE)
    return X_train, X_test, Y_train, Y_test


def get_advertisement_clicks_data_raw(train_size=0.8):
    assert train_size >= 0
    clics_df = pd.read_csv(get_data_dir('advertisement_clicks.csv'))
    X = clics_df['advertisement_id'].values
    Y = clics_df['action'].values

    X_train, X_test, Y_train, Y_test = split_by_train_size(X, Y, train_size, RANDOM_STATE)
    return X_train, X_test, Y_train, Y_test


