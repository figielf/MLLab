import numpy as np
import tensorflow as tf
from scipy.sparse import lil_matrix

from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2


class autorec_recommender:
    def __init__(self, K):
        self.K = K  # latent dim
        self.mu = None
        self.predict_model = None

    def fit(self, ratings_sparse_matrix, ratings_test_sparse_matrix, n_epochs=20, learing_rate=0.08, batch_size=128, reg=0.0001):
        N, M = ratings_sparse_matrix.shape  # N - number of users, M - number of items
        n_batches = N // batch_size + 1  # batches of users
        n_test_batches = ratings_test_sparse_matrix.shape[0] // batch_size + 1

        def train_generator(data_sparse, mask):
            while True:
                data_sparse, mask = shuffle(data_sparse, mask)
                for batch in range(n_batches):
                    up = (batch + 1) * batch_size
                    x = data_sparse[batch * batch_size:min(up, data_sparse.shape[0])].toarray()
                    x_mask = mask[batch * batch_size:min(up, data_sparse.shape[0])].toarray()
                    x = x - x_mask * self.mu
                    x_target = x
                    yield x, x_target

        def test_generator(data_sparse, mask, test_data_sparse, test_mask):
            while True:
                for batch in range(n_batches):
                    up = (batch + 1) * batch_size
                    x_test = data_sparse[batch * batch_size:min(up, data_sparse.shape[0])].toarray()
                    x_test_mask = mask[batch * batch_size:min(up, data_sparse.shape[0])].toarray()
                    x_test = x_test - x_test_mask * self.mu

                    x_target = test_data_sparse[batch * batch_size:min(up, data_sparse.shape[0])].toarray()
                    x_target_mask = test_mask[batch * batch_size:min(up, data_sparse.shape[0])].toarray()
                    x_target = x_target - x_target_mask * self.mu
                    yield x_test, x_target

        def sparse_data_mse(y_true, y_pred):
            target_mask = tf.cast(tf.math.not_equal(y_true, 0), dtype='float32')
            errors = y_pred - y_true
            squared_masked_errors = target_mask * errors * errors
            sse = tf.math.reduce_sum(tf.math.reduce_sum(squared_masked_errors))
            mse = sse / tf.math.reduce_sum(tf.math.reduce_sum(target_mask))
            return mse

        ratings_data = ratings_sparse_matrix.copy()
        ratings_test_data = ratings_test_sparse_matrix.copy()
        train_mask = (ratings_data > 0) * 1
        test_mask = (ratings_test_data > 0) * 1
        self.mu = ratings_data.sum() / train_mask.sum()  # ratings global bias

        i = Input(shape=(M, ))  # output shape=(N, M)
        x = Dropout(0.7)(i)  # output shape=(N, M)
        x = Dense(700, activation='tanh', kernel_regularizer=l2(reg))(x)  # output shape=(N, 700)
        x = Dense(M, kernel_regularizer=l2(reg))(x)  # output shape=(N, M)

        model = Model(inputs=i, outputs=x)
        model.compile(loss=sparse_data_mse, optimizer=SGD(learning_rate=learing_rate, momentum=0.9), metrics=[sparse_data_mse])
        print(model.summary())

        self.predict_model = Model(inputs=i, outputs=x)

        history = model.fit(
            train_generator(ratings_sparse_matrix, train_mask),
            validation_data=test_generator(ratings_sparse_matrix, train_mask, ratings_test_sparse_matrix, test_mask),
            epochs=n_epochs,
            #batch_size=128,
            steps_per_epoch=n_batches,
            validation_steps=n_test_batches
            )

        self.predict_model = Model(inputs=i, outputs=x)
        return history

    def predict(self, x_sparse):
        r_hat = self.predict_model.predict(x_sparse.toarray()) + self.mu
        r_hat = np.maximum(r_hat, 0.5)
        r_hat = np.minimum(r_hat, 5.0)
        return r_hat
