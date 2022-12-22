import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from tests.utils.data_utils import get_data_dir


def get_movielens_20m_sparse_data_raw():
    print('Reading raw movielens-20m-dataset data...')
    df = pd.read_csv(get_data_dir(os.path.join('large_files', 'movielens-20m-dataset', 'rating.csv')))
    return df


def get_movielens_20m_sparse_data_reindexed(remove_columns=True):
    t0 = datetime.now()
    df = get_movielens_20m_sparse_data_raw()
    print('Reindexing movielens-20m-dataset data...')

    assert len(set(df['userId'].values)) == np.max(df['userId'].values)
    df['user_idx'] = df['userId'] - 1

    movies = set(df['movieId'].values)
    movie2idx_map = {m: i for i, m in enumerate(movies)}
    df['movie_idx'] = df['movieId'].apply(lambda row: movie2idx_map[row])

    if remove_columns:
        df.drop(columns=['timestamp', 'userId', 'movieId'], inplace=True)

    print(f'Finished movielens-20m-dataset data reindexing in {datetime.now()-t0} ...')
    return df


def get_movielens_20m_sparse_data_sorted(n_top_users=None, n_top_movies=None, remove_columns=True):
    t0 = datetime.now()
    df = get_movielens_20m_sparse_data_raw()
    if remove_columns:
        df.drop(columns=['timestamp'], inplace=True)

    print(f'Filtering movielens-20m-dataset data by top {n_top_users} users and top {n_top_movies} movies...')

    user_ratings_counter = Counter(df['userId'])
    user_ratings_sorted = user_ratings_counter.most_common(n_top_users)
    top_users = {id_: new_idx for new_idx, (id_, count) in enumerate(user_ratings_sorted)}

    movie_ratings_counter = Counter(df['movieId'])
    movie_ratings_sorted = movie_ratings_counter.most_common(n_top_movies)
    top_movies = {id_: new_idx for new_idx, (id_, count) in enumerate(movie_ratings_sorted)}

    df_filtered = df[df['userId'].isin(top_users.keys()) & df['movieId'].isin(top_movies.keys())].copy()
    print(f'Finished filtering data in {datetime.now() - t0}...')

    print(f'Reindexing data...')
    t1 = datetime.now()
    df_filtered['user_idx'] = -1
    df_filtered['movie_idx'] = -1
    for id_, new_idx in top_users.items():
        df_filtered.loc[df_filtered['userId'] == id_, 'user_idx'] = new_idx
    for id_, new_idx in top_movies.items():
        df_filtered.loc[df_filtered['movieId'] == id_, 'movie_idx'] = new_idx

    cols = df_filtered.columns.tolist()
    df_filtered = df_filtered.reindex(columns=['user_idx', 'movie_idx'] + cols[:-2])

    print(f'Finished reindexing data in {datetime.now() - t1}...')

    if remove_columns:
        df_filtered.drop(columns=['userId', 'movieId'], inplace=True)

    print(f'Finished movielens-20m-dataset data reindexing in {datetime.now() - t0} ...')
    return df_filtered