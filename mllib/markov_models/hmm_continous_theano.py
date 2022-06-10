from datetime import datetime

import numpy as np
import theano
import theano.tensor as T


class hmm_continous_theano:
    def __init__(self, m, k, use_softmax=False):
        self.cost_op = None
        self.M = m  # number of hidden states
        self.K = k # number of Gaussians
        self.pi = None
        self.A = None
        self.R = None
        self.mu0 = None
        self.sigma0 = None
        self.use_softmax = use_softmax

    def fit(self, x, learning_rate=0.001, max_iter=10, initial_probs=None, random_state=123):
        # use_softmax=True set pi, A and B to be presoftmax args so that will not stand for probability distribution,
        # will have to be softmax_ed to become probability distribution
        # train the HMM model using the Baum-Welch algorithm (a kind of the expectation-maximization algorithm)
        t0 = datetime.now()

        N = len(x)
        D = x[0].shape[1]  # assume each x is organized (T, D)

        if initial_probs is not None:
            pi0, A0, R0, mu0, sigma0 = initial_probs
        else:
            # initialize distributions randomly
            np.random.seed(random_state)
            pi0 = np.ones(self.M) / self.M  # initial state distribution
            A0 = self._random_normalized(self.M, self.M)  # state transition matrix
            R0 = np.ones((self.M, self.K)) / self.K  # mixture proportions
            mu0 = np.zeros((self.M, self.K, D))
            for i in range(self.M):
                for k in range(self.K):
                    random_idx = np.random.choice(N)
                    x_rand = x[random_idx]
                    random_time_idx = np.random.choice(len(x_rand))
                    mu0[i, k] = x_rand[random_time_idx]
            sigma0 = np.zeros((self.M, self.K, D, D))
            for j in range(self.M):
                for k in range(self.K):
                    sigma0[j, k] = np.eye(D)

        x_seq, cost = self.set(pi0, A0, R0, mu0, sigma0)

        pi_update = self.pi - learning_rate * T.grad(cost, self.pi)
        A_update = self.A - learning_rate * T.grad(cost, self.A)
        R_update = self.R - learning_rate * T.grad(cost, self.R)
        mu_update = self.mu - learning_rate * T.grad(cost, self.mu)
        sigma_update = self.sigma - learning_rate * T.grad(cost, self.sigma)

        if self.use_softmax is False:
            # because we do not use softmax we have to normalise so that it is probability distribution (it sums to 1)
            pi_update = pi_update / pi_update.sum()
            A_update = A_update / A_update.sum(axis=1).dimshuffle(0, 'x')
            R_update = R_update / R_update.sum(axis=1).dimshuffle(0, 'x')

        updates = [
            (self.pi, pi_update),
            (self.A, A_update),
            (self.R, R_update),
            (self.mu, mu_update),
            (self.sigma, sigma_update),
        ]

        train_op = theano.function(
            inputs=[x_seq],
            updates=updates,
            allow_input_downcast=True,
        )

        history = []
        for step in range(max_iter):
            #if step % 10 == 0:
            print('step:', step)

            for n in range(N):
                c = np.sum(self.sequence_log_likelihood(x))
                history.append(c)
                train_op(x[n])

        print("Discrete HMM fit duration:", (datetime.now() - t0))
        return history, (self.pi.get_value(), self.A.get_value(), self.R.get_value(), self.mu.get_value(), self.sigma.get_value())

    def _random_normalized(self, d1, d2):
        x = np.random.random((d1, d2))
        return x / x.sum(axis=1, keepdims=True)

    def set(self, pi, A, R, mu, sigma):
        self.pi = theano.shared(pi)
        self.A = theano.shared(A)
        self.R = theano.shared(R)
        self.mu = theano.shared(mu)
        self.sigma = theano.shared(sigma)

        D = self.mu.shape[2]

        if self.use_softmax is True:
            pi_distr = T.nnet.softmax(self.pi).flatten()  # softmax returns 1xD if input is a 1-D array of size D
            A_distr = T.nnet.softmax(self.A)
            R_distr = T.nnet.softmax(self.R)
        else:
            pi_distr = self.pi
            A_distr = self.A
            R_distr = self.R

        def mvn_pdf(x, m, S):
            k = 1 / T.sqrt((2 * np.pi) ** D * T.nlinalg.det(S))
            e = T.exp(-0.5 * (x - m).T.dot(T.nlinalg.matrix_inverse(S).dot(x - m)))
            return k * e

        x_seq = T.matrix('x_seq')
        def gmm_pdf(x_seq):
            def state_pdf(xt):
                def comp_pdf(j, xt):
                    Bt_j = 0
                    for k in range(self.K):
                        S = self.sigma[j, k]
                        S_2 = S.dot(S.T)
                        Bt_j +=  R_distr[j, k] * mvn_pdf(xt, self.mu[j, k], S_2)
                    return Bt_j

                Bt = theano.scan(
                    fn=comp_pdf,
                    sequences=T.arange(self.M),
                    outputs_info=None,
                    n_steps=self.M,
                    non_sequences=[xt]
                )
                return Bt

            B, _ = theano.scan(
                fn=state_pdf,
                sequences=x_seq,
                outputs_info=None,
                n_steps=x_seq.shape[0]
            )

            return B.T

        B = gmm_pdf(x_seq)

        def cals_prob_step(t, prev_alpha, B):
            alpha = B[:, t] * prev_alpha.dot(A_distr)
            s = alpha.sum()
            return alpha / s, s

        [alpha, scale], _ = theano.scan(
            fn=cals_prob_step,
            sequences=T.arange(1, x_seq.shape[0]),
            outputs_info=[pi_distr * B[:, 0], None],
            n_steps=x_seq.shape[0] - 1,
            non_sequences=[B],
        )

        cost = -T.log(scale).sum()
        self.cost_op = theano.function(
            inputs=[x_seq],
            outputs=cost,
        )
        return x_seq, cost


    def forward_scaled(self, x):
        # first part of the hmm forward-backward algorithm
        # x - observed sequence, z - is not observable hidden states sequence
        # returns probability of sequence under model parameters = likelihood of one sequence sequence observation
        return self.cost_op(x)

    def sequence_log_likelihood(self, x):
        return [self.forward_scaled(xi).item() for xi in x]

    def log_likelihood(self, x):
        return self.forward_scaled(x)
