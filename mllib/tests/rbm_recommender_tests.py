import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix, save_npz

from recommenders.rbm_recommender import rbm_recommender
from tests.colaborative_filtering_tests import preprocess
from tests.matrix_factorization_and_deep_learning_recommender_tests import load_data_as_sparse_matrices

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


if __name__ == '__main__':
    n_top_users = 1000
    n_top_movies = 200
    data_train, data_test, u2m, m2u, um2r, um2r_test = preprocess(n_top_users, n_top_movies, train_size=0.8,
                                                                  load_mode=True)
    print('train data shape:', data_train.shape)
    print('test data shape:', data_test.shape)
    print('example of data:', data_train.head())
    plt.figure(figsize=(16, 16))

    print('\ntrain rbm autoencoder learning model')
    print('processing data as scipy sparse matrices ...')
    data_train_sparse, data_test_sparse = load_data_as_sparse_matrices(data_train, data_test)
    print(data_train_sparse.shape)
    print('finished processing data to sparse matrices')
    rbm_model = rbm_recommender(n_items=n_top_movies, hidden_layer_size=50, n_rating_classes=10)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        session.run(init_op)
        rbm_model.set_session(session)
        costs_rbm = rbm_model.fit(data_train_sparse, data_test_sparse, n_epochs=30, batch_size=256)

    plt.plot(costs_rbm['MSE_train'], label='rbm train MSE error')
    plt.plot(costs_rbm['MSE_test'], label='rbm test MSE error')

    plt.legend()
    plt.show()
