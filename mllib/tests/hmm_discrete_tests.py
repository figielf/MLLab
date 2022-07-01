import numpy as np
from matplotlib import pyplot as plt

from tests.utils.data_utils import get_coin_flip_data
from markov_models.hmm_discrete import hmm_discrete
from markov_models.hmm_discrete_theano import hmm_discrete_theano


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

def fit_coin_test_theano(hmm_model, x, fit_method, probs_init):
    history, params = fit_method(x, initial_probs=probs_init, max_iter=5)
    #plt.plot(history['cost'])
    #plt.show()

    print("fitted pi:", params[0])
    print("fitted A:", params[1])
    print("fitted B:", params[2])


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

    # pure vetribi algorithm without scaling
    hmm_vetribi = hmm_discrete(M)
    fit_coin_test(hmm_vetribi, X_coins, hmm_vetribi.fit, (PI.copy(), A.copy(), B.copy()))
    L = np.sum(hmm_vetribi.sequence_log_likelihood(X_coins))
    print("LL with fitted params:", L, '\n')

    # vetribi algorithm with scaling
    hmm_vetribi_scaled = hmm_discrete(M)
    fit_coin_test(hmm_vetribi_scaled, X_coins, hmm_vetribi_scaled.fit_scaled, (PI.copy(), A.copy(), B.copy()))
    L = np.sum(hmm_vetribi_scaled.sequence_log_likelihood_scaled(X_coins))
    print("LL with fitted with scaling params:", L, '\n')

    # automatic gradient descent by theano
    hmm_theano = hmm_discrete_theano(2)
    fit_coin_test_theano(hmm_theano, X_coins, hmm_theano.fit, (PI.copy(), A.copy(), B.copy()))
    L = np.sum(hmm_theano.sequence_log_likelihood(X_coins))
    print("LL with fitted params optimised with gradient descent by theano :", L, '\n')

    # automatic gradient descent by theano with softmax
    hmm_theano2 = hmm_discrete_theano(2, use_softmax=True)
    fit_coin_test_theano(hmm_theano2, X_coins, hmm_theano2.fit, (PI.copy(), A.copy(), B.copy()))
    L = np.sum(hmm_theano2.sequence_log_likelihood(X_coins))
    print("LL with fitted params optimised with gradient descent by theano with softmax :", L, '\n')

    # true variables
    PI = np.array([0.5, 0.5])
    A = np.array([[0.1, 0.9], [0.8, 0.2]])
    B = np.array([[0.6, 0.4], [0.3, 0.7]])
    print("\ntrue pi:", PI)
    print("true A:", A)
    print("true B:", B)

    hmm_vetribi2 = hmm_discrete(M)
    hmm_vetribi2.pi = PI.copy()
    hmm_vetribi2.A = A.copy()
    hmm_vetribi2.B = B.copy()
    print('LL with true params:', np.sum(hmm_vetribi2.sequence_log_likelihood_scaled(X_coins)))
