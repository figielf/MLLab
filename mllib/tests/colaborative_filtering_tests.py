import os
import pickle

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from recommenders.collaborative_filtering import collaborative_filtering
from tests.utils.data_utils import get_data_dir
from tests.utils.recommenders_data_utils import get_movielens_20m_sparse_data_sorted


def index_reset(df):
    df = df.reset_index()
    return df.drop(columns=['index'])


def process_data(n_top_users, n_top_movies, train_size=None):
    data = get_movielens_20m_sparse_data_sorted(n_top_users, n_top_movies, remove_columns=True)

    save_folder_path = get_data_dir(os.path.join('large_files', 'movielens-20m-dataset'))
    save_file_name = f'movielens_20m_sparse_{n_top_users}_topusers_{n_top_movies}_topmovies.csv'
    save_file_path = os.path.join(save_folder_path, save_file_name)

    if train_size is None:
        data_shuffled = shuffle(data)
        data_shuffled.to_csv(save_file_path, index=False)
        print(f'Filtered movielens data saved into {save_file_path}')
        data_shuffled = index_reset(data_shuffled)
        return data_shuffled, None
    else:
        data_train, data_test = train_test_split(data, train_size)
        data_train.to_csv(save_file_path, index=False)
        print(f'Filtered movielens train data saved into {save_file_path}')
        save_test_file_name = f'movielens_20m_sparse_{n_top_users}_topusers_{n_top_movies}_topmovies_test.csv'
        save_test_file_path = os.path.join(save_folder_path, save_test_file_name)
        data_test.to_csv(save_test_file_path, index=False)
        print(f'Filtered movielens test data saved into {save_test_file_path}')
        data_train = index_reset(data_train)
        data_test = index_reset(data_test)
        return data_train, data_test


def load_data(n_top_users, n_top_movies, include_test_data=True):
    save_folder_path = get_data_dir(os.path.join('large_files', 'movielens-20m-dataset'))
    save_file_name = f'movielens_20m_sparse_{n_top_users}_topusers_{n_top_movies}_topmovies.csv'
    save_file_path = os.path.join(save_folder_path, save_file_name)

    save_test_file_name = f'movielens_20m_sparse_{n_top_users}_topusers_{n_top_movies}_topmovies_test.csv'
    save_test_file_path = os.path.join(save_folder_path, save_test_file_name)

    data_train = pd.read_csv(save_file_path)
    data_train = index_reset(data_train)
    print(f'movielens data loaded from {save_test_file_path}')
    if include_test_data:
        data_test = pd.read_csv(save_test_file_path)
        print(f'movielens test data loaded from {save_test_file_path}')
        data_test = index_reset(data_test)
    else:
        data_test = None
    return data_train, data_test


def train_test_split(df, train_size):
    df_shuffled = shuffle(df)
    cutoff = int(train_size * len(df))
    df_train = df_shuffled.iloc[:cutoff].copy()
    df_test = df_shuffled.iloc[cutoff:].copy()
    return df_train, df_test


def extract_dictionaries(df_train, df_test, n_top_users, n_top_movies):
    user2movie = {}
    movie2user = {}
    usermovie2rating = {}
    usermovie2rating_test = {}

    def update_user2movie_and_movie2user(row):
        i = int(row['user_idx'])
        j = int(row['movie_idx'])
        if i not in user2movie:
            user2movie[i] = [j]
        else:
            user2movie[i].append(j)

        if j not in movie2user:
            movie2user[j] = [i]
        else:
            movie2user[j].append(i)

        usermovie2rating[(i, j)] = row['rating']

    def update_usermovie2rating_test(row):
        i = int(row['user_idx'])
        j = int(row['movie_idx'])
        usermovie2rating_test[(i, j)] = row['rating']

    print('Calling: update_user2movie_and_movie2user')
    df_train.apply(update_user2movie_and_movie2user, axis=1)

    print('Calling: update_usermovie2rating_test')
    df_test.apply(update_usermovie2rating_test, axis=1)

    print('Saving dictionaries...')
    save(user2movie, f'user2movie_{n_top_users}_topusers_{n_top_movies}_topmovies.json')
    save(movie2user, f'movie2user_{n_top_users}_topusers_{n_top_movies}_topmovies.json')
    save(usermovie2rating, f'usermovie2rating_{n_top_users}_topusers_{n_top_movies}_topmovies.json')
    save(usermovie2rating_test, f'usermovie2rating_test_{n_top_users}_topusers_{n_top_movies}_topmovies.json')

    return user2movie, movie2user, usermovie2rating, usermovie2rating_test


def save(data, file_name):
    save_folder_path = get_data_dir(os.path.join('large_files', 'movielens-20m-dataset'))
    save_file_path = os.path.join(save_folder_path, file_name)
    with open(save_file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Data saved into {save_file_path}')


def load_dictionaries(n_top_users, n_top_movies):
    file_name_suffix = f'{n_top_users}_topusers_{n_top_movies}_topmovies.json'
    user2movie = load('user2movie_' + file_name_suffix)
    movie2user = load('movie2user_' + file_name_suffix)
    usermovie2rating = load('usermovie2rating_' + file_name_suffix)
    usermovie2rating_test = load('usermovie2rating_test_' + file_name_suffix)
    return user2movie, movie2user, usermovie2rating, usermovie2rating_test


def load(file_name):
    load_folder_path = get_data_dir(os.path.join('large_files', 'movielens-20m-dataset'))
    load_file_path = os.path.join(load_folder_path, file_name)
    with open(load_file_path, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded data from {load_file_path}')
    return data


def preprocess(n_top_users, n_top_movies, train_size=0.8, load_mode=True):
    if load_mode:
        data_train, data_test = load_data(n_top_users, n_top_movies)
        u2m, m2u, um2r, um2r_test = load_dictionaries(n_top_users, n_top_movies)
    else:
        data_train, data_test = process_data(n_top_users, n_top_movies, train_size=train_size)
        u2m, m2u, um2r, um2r_test = extract_dictionaries(data_train, data_test, n_top_users, n_top_movies)

    return data_train, data_test, u2m, m2u, um2r, um2r_test


def predict(user_item2rating, model):
    predictions = []
    targets = []
    for (u, i), target in user_item2rating.items():
        prediction = model.predict(u, i)
        predictions.append(prediction)
        targets.append(target)
    return predictions, targets


def mse(p, t):
    return np.mean((np.array(p) - np.array(t)) ** 2)


if __name__ == '__main__':
    n_top_users = 1000
    n_top_movies = 200
    data_train, data_test, u2m, m2u, um2r, um2r_test = preprocess(n_top_users, n_top_movies, train_size=0.8,
                                                                  load_mode=True)
    print('train data shape:', data_train.shape)
    print('test data shape:', data_test.shape)

    # user-user collaborative filter
    print('\nuser-user collaborative filter')
    user_uswr_model = collaborative_filtering(k_neighbours=25, min_common_items=5)
    user_uswr_model.fit(user2item=u2m, item2user=m2u, user_item2rating=um2r)

    train_predictions, train_targets = predict(um2r, user_uswr_model)
    test_predictions, test_targets = predict(um2r_test, user_uswr_model)

    print('train mse:', mse(train_predictions, train_targets))
    print('test mse:', mse(test_predictions, test_targets))



    # item-item collaborative filter
    print('\nitem-item collaborative filter')
    mu2r = {(m, u): r for (u, m), r in um2r.items()}
    mu2r_test = {(m, u): r for (u, m), r in um2r_test.items()}

    item_item_model = collaborative_filtering(k_neighbours=20, min_common_items=5)
    item_item_model.fit(user2item=m2u, item2user=u2m, user_item2rating=mu2r)

    train_predictions, train_targets = predict(mu2r, item_item_model)
    test_predictions, test_targets = predict(mu2r_test, item_item_model)

    print('train mse:', mse(train_predictions, train_targets))
    print('test mse:', mse(test_predictions, test_targets))
