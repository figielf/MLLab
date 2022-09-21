import numpy as np
from scipy.stats import norm, t


def z_confidence_intervals(x, mu, sigma=None, alpha=0.95):
    n = len(x)
    if sigma is not None:
        sigma_hat = sigma
    else:
        sigma_hat = np.std(x, ddof=1)

    p = (1 - alpha) / 2
    standard_left = norm.ppf(p)
    standard_right = norm.ppf(1 - p)

    lower = mu + standard_left * sigma_hat / np.sqrt(n)
    upper = mu + standard_right * sigma_hat / np.sqrt(n)
    return lower, upper


def t_confidence_intervals(x, mu, alpha=0.95):
    n = len(x)
    sigma_hat = np.std(x, ddof=1)

    p = (1 - alpha) / 2
    standard_left = t.ppf(p, df=n - 1)
    standard_right = t.ppf(1 - p, df=n - 1)

    lower = mu + standard_left * sigma_hat / np.sqrt(n)
    upper = mu + standard_right * sigma_hat / np.sqrt(n)
    return lower, upper
