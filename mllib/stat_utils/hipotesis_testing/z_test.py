import numpy as np
from scipy.stats import norm


def z_test(x, y=None, mu0=0, sigma_x=None, sigma_y=None, alternative='two-sided'):
    assert alternative in ['two-sided', 'smaller', 'larger'], \
        "alternative has be be one of ['two-sided', 'smaller', 'larger']"

    n_x = len(x)
    x_mu = np.mean(x)
    if sigma_x is not None:
        x_sigma2_hat = sigma_x ** 2
    else:
        x_sigma2_hat = np.var(x, ddof=1)

    if y is None:
        mu_hat = x_mu
        mu_hat_var = x_sigma2_hat / n_x
    else:
        n_y = len(y)
        y_mu = np.mean(y)
        if sigma_y is not None:
            y_sigma2_hat = sigma_y ** 2
        else:
            y_sigma2_hat = np.var(y, ddof=1)

        mu_hat = x_mu - y_mu
        mu_hat_var = x_sigma2_hat / n_x + y_sigma2_hat / n_y

    z_statistics = (mu_hat - mu0) / np.sqrt(mu_hat_var)

    if alternative == 'two-sided':
        p_value_right = 1 - norm.cdf(np.abs(z_statistics))
        p_value_left = norm.cdf(-np.abs(z_statistics))
        p_value = p_value_left + p_value_right
    elif alternative == 'smaller':
        p_value = norm.cdf(z_statistics)
    else:  # alternative == 'larger'
        p_value = 1 - norm.cdf(z_statistics)

    return z_statistics, p_value
