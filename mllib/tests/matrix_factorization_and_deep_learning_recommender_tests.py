import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix, save_npz

from recommenders.autorec_recommender import autorec_recommender
from recommenders.deep_learning_recommender import deep_learning_recommender
from recommenders.deep_residual_network_recommender import deep_residual_network_recommender
from recommenders.matrix_factorization_recommender import matrix_factorization_recommender
from recommenders.rbm_recommender import rbm_recommender
from tests.colaborative_filtering_tests import preprocess, predict, mse


def predict_from_tf2_model(user_item2rating, model):
    users = []
    items = []
    targets = []
    for (u, i), target in user_item2rating.items():
        users.append(u)
        items.append(i)
        targets.append(target)
    predictions = model.predict(np.array(users), np.array(items))
    return predictions.flatten(), np.array(targets)


def load_data_as_sparse_matrices(df_train, df_test, save_sparce_matrices=False):
    def update_train(row):
        i = int(row['user_idx'])
        j = int(row['movie_idx'])
        train_data_sparse[i, j] = row['rating']

    def update_test(row):
        i = int(row['user_idx'])
        j = int(row['movie_idx'])
        test_data_sparse[i, j] = row['rating']

    N = max([df_train['user_idx'].max(), df_test['user_idx'].max()]) + 1  # number of users
    M = max([df_train['movie_idx'].max(), df_test['movie_idx'].max()]) + 1  # number of users

    train_data_sparse = lil_matrix((N, M))
    test_data_sparse = lil_matrix((N, M))

    print("Calling: update_train")
    df_train.apply(update_train, axis=1)
    train_data_sparse = train_data_sparse.tocsr()

    # test ratings dictionary
    print("Calling: update_test")
    df_test.apply(update_test, axis=1)
    test_data_sparse = test_data_sparse.tocsr()

    if save_sparce_matrices:
        save_npz('movielens_train_data_sparse.npz', train_data_sparse)
        save_npz('movielens_test_data_sparse.npz', test_data_sparse)
    return train_data_sparse, test_data_sparse


if __name__ == '__main__':
    n_top_users = 1000
    n_top_movies = 200
    data_train, data_test, u2m, m2u, um2r, um2r_test = preprocess(n_top_users, n_top_movies, train_size=0.8,
                                                                  load_mode=True)
    print('train data shape:', data_train.shape)
    print('test data shape:', data_test.shape)
    print('example of data:', data_train.head())
    plt.figure(figsize=(16, 16))
    #
    # print('\ntrain matrix factorization model by alternating least squares')
    # mf_model_als = matrix_factorization_recommender(K=10)
    # costs_als = mf_model_als.fit_by_als(user2item=u2m, item2user=m2u, user_item2rating=um2r,
    #                                     user_item2rating_test=um2r_test, n_epochs=25, reg=20.)
    #
    # train_predictions_als, train_targets_als = predict(um2r, mf_model_als)
    # test_predictions_als, test_targets_als = predict(um2r_test, mf_model_als)
    # print('final train mse:', mse(train_predictions_als, train_targets_als))
    # print('final test mse:', mse(test_predictions_als, test_targets_als))
    #
    # plt.plot(costs_als['train_cost'], label='mf als train cost')
    # plt.plot(costs_als['test_cost'], label='mf als test cost')
    #
    # print('\ntrain matrix factorization model with tensorflow 2')
    # mf_model_tf2 = matrix_factorization_recommender(K=10)
    # costs_tf2 = mf_model_tf2.fit_by_tf2(data_train, data_test, 'user_idx', 'movie_idx', 'rating', n_epochs=25, reg=0.)
    #
    # train_predictions_tf2, train_targets_tf2 = predict(um2r, mf_model_tf2)
    # test_predictions_tf2, test_targets_tf2 = predict(um2r_test, mf_model_tf2)
    # print('final train mse:', mse(train_predictions_tf2, train_targets_tf2))
    # print('final test mse:', mse(test_predictions_tf2, test_targets_tf2))
    #
    # plt.plot(costs_tf2.history['loss'], label='mf tf2 train regularized cost')
    # plt.plot(costs_tf2.history['val_loss'], label='mf tf2 test regularized cost')
    # plt.plot(costs_tf2.history['mse'], label='mf tf2 train cost')
    # plt.plot(costs_tf2.history['val_mse'], label='mf tf2 test cost')
    #
    # print('\ntrain deep learning model')
    # dl_model = deep_learning_recommender(K=10)
    # costs_dl = dl_model.fit(data_train, data_test, 'user_idx', 'movie_idx', 'rating', n_epochs=25, reg=0.)
    #
    # train_predictions_dl, train_targets_dl = predict_from_tf2_model(um2r, dl_model)
    # test_predictions_dl, test_targets_dl = predict_from_tf2_model(um2r_test, dl_model)
    # print('final train mse:', mse(train_predictions_dl, train_targets_dl))
    # print('final test mse:', mse(test_predictions_dl, test_targets_dl))
    #
    # plt.plot(costs_dl.history['loss'], label='deep learning train regularized cost')
    # plt.plot(costs_dl.history['val_loss'], label='deep learning test regularized cost')
    # plt.plot(costs_dl.history['mse'], label='deep learning train cost')
    # plt.plot(costs_dl.history['val_mse'], label='deep learning test cost')
    #
    # print('\ntrain residual learning model')
    # rl_model = deep_residual_network_recommender(K=10)
    # costs_rl = rl_model.fit(data_train, data_test, 'user_idx', 'movie_idx', 'rating', n_epochs=25, reg=0.)
    #
    # train_predictions_rl, train_targets_rl = predict_from_tf2_model(um2r, rl_model)
    # test_predictions_rl, test_targets_rl = predict_from_tf2_model(um2r_test, rl_model)
    # print('final train mse:', mse(train_predictions_rl, train_targets_rl))
    # print('final test mse:', mse(test_predictions_rl, test_targets_rl))
    #
    # plt.plot(costs_rl.history['loss'], label='residual learning train regularized cost')
    # plt.plot(costs_rl.history['val_loss'], label='residual learning test regularized cost')
    # plt.plot(costs_rl.history['mse'], label='residual learning train cost')
    # plt.plot(costs_rl.history['val_mse'], label='residual learning test cost')

    print('\ntrain autorec autoencoder learning model')
    print('processing data as scipy sparse matrices ...')
    data_train_sparse, data_test_sparse = load_data_as_sparse_matrices(data_train, data_test)
    print('finished processing data to sparse matrices')
    auto_model = autorec_recommender(K=10)
    costs_auto = auto_model.fit(data_train_sparse, data_test_sparse, n_epochs=30, reg=0.0001)

    plt.plot(costs_auto.history['loss'], label='autorec train regularized cost')
    plt.plot(costs_auto.history['val_loss'], label='autorec test regularized cost')
    plt.plot(costs_auto.history['sparse_data_mse'], label='autorec train cost')
    plt.plot(costs_auto.history['val_sparse_data_mse'], label='autorec test cost')

    plt.legend()
    plt.show()
