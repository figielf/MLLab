import pandas as pd

from tests.utils.data_utils import get_data_dir


def get_stock_market_timeseries_data(train_size=0.5):
    assert train_size >= 0
    df = pd.read_csv(get_data_dir('aapl_msi_sbux.csv')).values

    n_train = int(df.shape[0] * train_size)
    train_df = df[:n_train]
    test_df = df[n_train:]
    return train_df, test_df


def get_airline_passengers_timeseries_data(train_size=0.8, should_plot=True):
    assert train_size >= 0
    df = pd.read_csv(get_data_dir('airline_passengers.csv'), index_col='Month', parse_dates=True)
    df.index.freq = 'MS'

    if should_plot:
        df['Passengers'].plot(figsize=(20, 8))

    n_train = int(df.shape[0] * train_size)
    train_df = df[:n_train]
    if n_train < df.shape[0]:
        test_df = df[n_train:]
    else:
        test_df = None
    return train_df, test_df


def get_sp500_timeseries_data(train_size=0.8):
    assert train_size >= 0
    df = pd.read_csv(get_data_dir('sp500_close.csv'), index_col=0, parse_dates=True)
    #df.index.freq = 'B'

    n_train = int(df.shape[0] * train_size)
    train_df = df[:n_train]
    if n_train < df.shape[0]:
        test_df = df[n_train:]
    else:
        test_df = None
    return train_df, test_df


def get_monthly_champagne_timeseries_data(train_size=0.8, should_plot=True):
    assert train_size >= 0
    df = pd.read_csv(get_data_dir('perrin-freres-monthly-champagne.csv'), index_col='Month', skipfooter=2, parse_dates=True)
    df.columns = ['Sales']
    df.index.freq = 'MS'

    if should_plot:
        df['Sales'].plot(figsize=(20, 8))

    n_train = int(df.shape[0] * train_size)
    train_df = df[:n_train]
    if n_train < df.shape[0]:
        test_df = df[n_train:]
    else:
        test_df = None
    return train_df, test_df
