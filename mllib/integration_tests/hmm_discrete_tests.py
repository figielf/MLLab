import numpy as np
from matplotlib import pyplot as plt

from integration_tests.utils.data_utils import get_coin_flip_data
from markov_models.hmm_discrete import hmm_discrete


def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


def fit_coin_test(hmm_model, x, fit_method, probs_init):
    history = fit_method(x, initial_probs=probs_init)
    plt.plot(history['cost'])
    plt.show()

    print("fitted pi:", hmm_model.pi)
    print("fitted A:", hmm_model.A)
    print("fitted B:", hmm_model.B)


if __name__ == '__main__':
    X_coins = get_coin_flip_data()
    M = 2  # number of hidden states
    V = max(max(xi) for xi in X_coins) + 1  # number of observable states

    # random initial probs
    print('Random initial probabilities scenario')
    PI = np.ones(M) / M
    A = random_normalized(M, M)
    B = random_normalized(M, V)
    print("\ninitial pi:", PI)
    print("initial A:", A)
    print("initial B:", B, '\n')

    hmm = hmm_discrete(M)
    fit_coin_test(hmm, X_coins, hmm.fit, (PI.copy(), A.copy(), B.copy()))
    L = np.sum(hmm.sequence_log_likelihood(X_coins))
    print("LL with fitted params:", L, '\n')

    hmm_scaled = hmm_discrete(M)
    fit_coin_test(hmm_scaled, X_coins, hmm_scaled.fit_scaled, (PI.copy(), A.copy(), B.copy()))
    L = np.sum(hmm_scaled.sequence_log_likelihood_scaled(X_coins))
    print("LL with fitted with scaling params:", L, '\n')

    # true variables
    PI = np.array([0.5, 0.5])
    A = np.array([[0.1, 0.9], [0.8, 0.2]])
    B = np.array([[0.6, 0.4], [0.3, 0.7]])
    print("\ntrue pi:", PI)
    print("true A:", A)
    print("true B:", B)
