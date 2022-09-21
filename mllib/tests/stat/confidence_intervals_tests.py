import numpy as np

from stat_utils.hipotesis_testing.confidence_intervals import z_confidence_intervals, t_confidence_intervals

np.random.seed(1)


def run_experiment(sample_generator, mu):
    x = sample_generator(mu)
    x_mean = np.mean(x)

    left_z, right_z = z_confidence_intervals(x, mu)
    left_t, right_t = t_confidence_intervals(x, mu)
    return left_z < x_mean and x_mean < right_z, left_t < x_mean and x_mean < right_t


def repeat_experiment(sample_generator, mu, n_repeats):
    z_hits_history = []
    t_hits_history = []
    for k in range(n_repeats):
        z_hits, t_hits = run_experiment(sample_generator, mu)
        z_hits_history.append(z_hits)
        t_hits_history.append(t_hits)
    return np.mean(z_hits_history), np.mean(t_hits_history)


def sample_normal(n, mu, sigma):
    x = np.random.randn(n)
    return x * sigma + mu


def sample_exponential(n, mu):
    x = np.random.exponential(scale=mu, size=n)
    return x


if __name__ == '__main__':
    N = 1000
    mu = 5
    sigma = 2

    X = np.random.randn(N)
    X = X * sigma + mu

    mu_X = np.mean(X)

    X = np.random.randn(N) * sigma + mu
    left_z, right_z = z_confidence_intervals(X, mu)
    print(f'sample mean={mu_X} z confidence intervals: left={left_z}, right={right_z}')
    left_t, right_t = t_confidence_intervals(X, mu)
    print(f'sample mean={mu_X} t confidence intervals: left={left_t}, right={right_t}')

    n_times = 10000
    z_hits_ratio, t_hits_ratio = repeat_experiment(lambda mu: sample_normal(N, mu, sigma), mu, n_times)
    print(f'Run experiment with {n_times} normal samples of size {N}, hit z CI=: {z_hits_ratio}, hit t CI: {t_hits_ratio}')
    z_hits_ratio, t_hits_ratio = repeat_experiment(lambda mu: sample_exponential(N, mu), mu, n_times)
    print(f'Run experiment with {n_times} exponential samples of size {N}, hit z CI=: {z_hits_ratio}, hit t CI: {t_hits_ratio}')

    N = 10
    z_hits_ratio, t_hits_ratio = repeat_experiment(lambda mu: sample_normal(N, mu, sigma), mu, n_times)
    print(f'Run experiment with {n_times} normal samples of size {N}, hit z CI=: {z_hits_ratio}, hit t CI: {t_hits_ratio}')
    z_hits_ratio, t_hits_ratio = repeat_experiment(lambda mu: sample_exponential(N, mu), mu, n_times)
    print(f'Run experiment with {n_times} exponential samples of size {N}, hit z CI=: {z_hits_ratio}, hit t CI: {t_hits_ratio}')
