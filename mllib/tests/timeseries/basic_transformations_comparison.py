import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox

from tests.utils.timeseries_data_utils import get_airline_passengers_timeseries_data

if __name__ == '__main__':
    data, _ = get_airline_passengers_timeseries_data(train_size=1.0, should_plot=False)

    data['SqrtPassengers'] = np.sqrt(data['Passengers'])
    data['LogPassengers'] = np.log(data['Passengers'])
    data['BoxCoxPassengers'], _lambda = boxcox(data['Passengers'])

    cols_to_plot = [col for col in data.columns if 'Passengers' in col]

    n_rows_plot = 2
    n_cols_plot = len(cols_to_plot)
    plt.figure(figsize=(20, 10))
    for i, col_name in enumerate(cols_to_plot):
        if col_name == 'BoxCoxPassengers':
            lambda_suffix = f', lambda={_lambda}'
        else:
            lambda_suffix = None
        plt.subplot(n_rows_plot, n_cols_plot, i + 1)
        data[col_name].plot()
        plt.title(f'{col_name} data{lambda_suffix}')
        plt.subplot(n_rows_plot, n_cols_plot, n_cols_plot + i + 1)
        data[col_name].hist(bins=20)
        plt.title(f'{col_name} hist{lambda_suffix}')
    plt.show()
