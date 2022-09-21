import numpy as np

from stat_utils.hipotesis_testing.z_test import z_test
from tests.utils.data_utils import get_titanic_data_raw, get_advertisement_clicks_data_raw

np.random.seed(1)


def run_on_titanic_data():
    print('\nUse Z test on Titanic data')
    x, _, y, _ = get_titanic_data_raw(train_size=1)
    x_survived = x[y == 1]['Fare'].dropna().values
    x_drown = x[y == 0]['Fare'].dropna().values

    print(f'Average Fare price for those who survived was: {np.mean(x_survived)}')
    print(f'Average Fare price for those who not survived was: {np.mean(x_drown)}')

    z_stat, z_p = z_test(x_survived, x_drown, mu0=0)
    print(f'Titanic Fare price, H0: fare price of survived = fare price of not survived - Z statistic: {z_stat}, p value: {z_p}')


def run_on_advertisement_clicks_data():
    print('\nUse Z test on Titanic data')
    x, _, y, _ = get_advertisement_clicks_data_raw(train_size=1)
    x_A = y[x == 'A']
    x_B = y[x == 'B']

    print(f'Click rate average for advertisement A: {np.mean(x_A)}')
    print(f'Click rate average for advertisement B: {np.mean(x_B)}')

    z_stat, z_p = z_test(x_A, x_B, mu0=0)
    print(f'Click rate, H0: click rates are same for advertisements A and B - Z statistic: {z_stat}, p value: {z_p}')

    z_stat, z_p = z_test(x_A, x_B, mu0=0, alternative='smaller')
    print(f'Click rate, H0: click rates are same for advertisements A and B - Z statistic: {z_stat}, p value: {z_p}')


if __name__ == '__main__':
    N = 100
    mu0 = 0
    mu = 0.2
    sigma = 1
    x = np.random.randn(N) * sigma + mu

    # one sample tests
    # two-sided test
    z_stat, z_p = z_test(x, mu0=0)
    print(f'One-sample two-sided Z test, H0: mu0={0}, sample ~ N({mu, sigma**2}) - Z statistic: {z_stat}, p value: {z_p}')

    # one-sided tests
    z_stat, z_p = z_test(x, mu0=0, alternative='larger')
    print(f'One-sample one-sided Z test, H0: mu0<={0}, sample ~ N({mu, sigma**2}) - Z statistic: {z_stat}, p value: {z_p}')

    z_stat, z_p = z_test(x, mu0=0, alternative='smaller')
    print(f'One-sample one-sided Z test, H0: mu0>={0}, sample ~ N({mu, sigma**2}) - Z statistic: {z_stat}, p value: {z_p}')

    # two samples tests
    N1 = 500
    mu1 = 0.5
    sigma1 = 1
    x0 = x
    x1 = np.random.randn(N1) * sigma1 + mu1
    print(x0.shape)
    print(x1.shape)

    # two-sided test
    z_stat, z_p = z_test(x0, x1, mu0=0)
    print(f'Two-samples two-sided Z test, H0: mu0={0}, sample0 ~ N({mu, sigma**2}), sample1 ~ N({mu1, sigma1**2}) - Z statistic: {z_stat}, p value: {z_p}')

    # one-sided tests
    z_stat, z_p = z_test(x0, x1, mu0=0, alternative='larger')
    print(f'Two-samples one-sided Z test, H0: mu0<={0}, sample0 ~ N({mu, sigma**2}), sample1 ~ N({mu1, sigma1**2}) - Z statistic: {z_stat}, p value: {z_p}')

    z_stat, z_p = z_test(x0, x1, mu0=0, alternative='smaller')
    print(f'Two-samples one-sided Z test, H0: mu0>={0}, sample0 ~ N({mu, sigma**2}), sample1 ~ N({mu1, sigma1**2}) - Z statistic: {z_stat}, p value: {z_p}')

    run_on_titanic_data()

    run_on_advertisement_clicks_data()
