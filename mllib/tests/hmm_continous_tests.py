import numpy as np
from matplotlib import pyplot as plt

from tests.utils.data_utils import get_helloworld_data
from markov_models.hmm_continous_theano import hmm_continous_theano


def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


def simple_init():
    M = 1
    K = 1
    D = 1
    pi = np.array([1])
    A = np.array([[1]])
    R = np.array([[1]])
    mu = np.array([[[0]]])
    sigma = np.array([[[[1]]]])
    return M, K, D, pi, A, R, mu, sigma


def big_init():
    M = 5
    K = 3
    D = 2

    pi = np.array([1, 0, 0, 0, 0])  # initial state distribution

    A = np.array([
        [0.9, 0.025, 0.025, 0.025, 0.025],
        [0.025, 0.9, 0.025, 0.025, 0.025],
        [0.025, 0.025, 0.9, 0.025, 0.025],
        [0.025, 0.025, 0.025, 0.9, 0.025],
        [0.025, 0.025, 0.025, 0.025, 0.9],
    ])  # state transition matrix - likes to stay where it is

    R = np.ones((M, K)) / K  # mixture proportions

    mu = np.array([
        [[0, 0], [1, 1], [2, 2]],
        [[5, 5], [6, 6], [7, 7]],
        [[10, 10], [11, 11], [12, 12]],
        [[15, 15], [16, 16], [17, 17]],
        [[20, 20], [21, 21], [22, 22]],
    ])  # M x K x D

    sigma = np.zeros((M, K, D, D))
    for m in range(M):
        for k in range(K):
            sigma[m, k] = np.eye(D)
    return M, K, D, pi, A, R, mu, sigma


def get_signals(N=20, T=100, init=big_init):
    M, K, D, pi, A, R, mu, sigma = init()
    X = []
    for n in range(N):
        x = np.zeros((T, D))
        s = 0  # initial state is 0 since pi[0] = 1
        r = np.random.choice(K, p=R[s])  # choose mixture
        x[0] = np.random.multivariate_normal(mu[s][r], sigma[s][r])
        for t in range(1, T):
            s = np.random.choice(M, p=A[s])  # choose state
            r = np.random.choice(K, p=R[s])  # choose mixture
            x[t] = np.random.multivariate_normal(mu[s][r], sigma[s][r])
        X.append(x)
    return X


def real_signal():
    signal = get_helloworld_data()
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std()

    hmm = hmm_continous_theano(3, 3)
    # signal needs to be of shape N x T(n) x D
    history, _ = hmm.fit(signal.reshape(1, T, 1), learning_rate=2e-7, max_iter=20)
    plt.plot(history)
    plt.show()


def fake_signal():
    signals = get_signals()
    hmm = hmm_continous_theano(5, 3)
    history, _ = hmm.fit(signals, max_iter=3)
    plt.plot(history)
    plt.show()
    L = np.sum(hmm.sequence_log_likelihood(signals))
    print("LL for fitted params:", L)

    # test in actual params
    _, _, _, pi, A, R, mu, sigma = big_init()

    # turn these into their "pre-softmax" forms
    pi = np.log(pi)
    A = np.log(A)
    R = np.log(R)

    # decompose sigma using cholesky factorization
    sigma = np.linalg.cholesky(sigma)

    hmm.set(pi, A, R, mu, sigma)
    L = np.sum(hmm.sequence_log_likelihood(signals))
    print("LL for actual params:", L)


if __name__ == '__main__':
    real_signal()
    # fake_signal()
