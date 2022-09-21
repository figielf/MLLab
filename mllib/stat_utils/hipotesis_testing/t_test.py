import numpy as np

from scipy.stats import t


def t_test(x, y=None, mu0=0, alternative='two-sided'):
    assert alternative in ['two-sided', 'smaller', 'larger'], \
        "alternative has be be one of ['two-sided', 'smaller', 'larger']"

    n_x = len(x)
    x_mu = np.mean(x)
    x_sigma2_hat = np.var(x, ddof=1)

    if y is None:
        mu_hat = x_mu
        mu_hat_var = x_sigma2_hat / n_x
        df = n_x - 1
    else:
        n_y = len(y)
        y_mu = np.mean(y)
        y_sigma2_hat = np.var(y, ddof=1)

        mu_hat = x_mu - y_mu
        mu_hat_var = x_sigma2_hat / n_x + y_sigma2_hat / n_y
        df = n_x + n_y - 2

    t_statistics = (mu_hat - mu0) / np.sqrt(mu_hat_var)

    if alternative == 'two-sided':
        p_value_right = 1 - t.cdf(np.abs(t_statistics), df=df)
        p_value_left = t.cdf(-np.abs(t_statistics), df=df)
        p_value = p_value_left + p_value_right
    elif alternative == 'smaller':
        p_value = t.cdf(t_statistics, df=df)
    else:  # alternative == 'larger'
        p_value = 1 - t.cdf(t_statistics, df=df)

    return t_statistics, p_value
