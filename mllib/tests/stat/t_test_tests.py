import numpy as np

from stat_utils.hipotesis_testing.t_test import t_test

np.random.seed(1)


if __name__ == '__main__':
    N = 100
    mu0 = 0
    mu = 0.2
    sigma = 1
    x = np.random.randn(N) * sigma + mu

    # one sample tests
    # two-sided test
    t_stat, t_p = t_test(x, mu0=0)
    print(f'Two-sided T test H0: mu0={0}, sample ~ N({mu, sigma**2}) - T statistic: {t_stat}, p value: {t_p}')

    # one-sided tests
    t_stat, t_p = t_test(x, mu0=0, alternative='larger')
    print(f'One-sided T test, H0: mu0<={0}, sample ~ N({mu, sigma**2}) - Z statistic: {t_stat}, p value: {t_p}')

    t_stat, t_p = t_test(x, mu0=0, alternative='smaller')
    print(f'One-sided T test, H0: mu0>={0}, sample ~ N({mu, sigma**2}) - Z statistic: {t_stat}, p value: {t_p}')

    # two samples tests
    N1 = 200
    mu1 = 0.5
    sigma1 = 1
    x0 = x
    x1 = np.random.randn(N1) * sigma1 + mu1

    # two-sided test
    t_stat, t_p = t_test(x0, x1, mu0=0)
    print(f'Two-samples two-sided T test, H0: mu0={0}, sample0 ~ N({mu, sigma**2}), sample1 ~ N({mu1, sigma1**2}) - T statistic: {t_stat}, p value: {t_p}')

    # one-sided tests
    t_stat, t_p = t_test(x0, x1, mu0=0, alternative='larger')
    print(f'Two-samples one-sided T test, H0: mu0<={0}, sample0 ~ N({mu, sigma**2}), sample1 ~ N({mu1, sigma1**2}) - T statistic: {t_stat}, p value: {t_p}')

    t_stat, t_p = t_test(x0, x1, mu0=0, alternative='smaller')
    print(f'Two-samples one-sided T test, H0: mu0>={0}, sample0 ~ N({mu, sigma**2}), sample1 ~ N({mu1, sigma1**2}) - Ts statistic: {t_stat}, p value: {t_p}')
