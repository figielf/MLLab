from datetime import datetime

import numpy as np
import theano
import theano.tensor as T


class hmm_discrete_theano:
    def __init__(self, m, use_softmax=False):
        self.cost_op = None
        self.M = m  # number of hidden states
        self.pi = None
        self.A = None
        self.B = None
        self.use_softmax = use_softmax

    def fit(self, x, learning_rate=0.001, max_iter=10, initial_probs=None, random_state=123):
        # use_softmax=True set pi, A and B to be presoftmax args so that will not stand for probability distribution,
        # will have to be softmax_ed to become probability distribution
        # train the HMM model using the Baum-Welch algorithm (a kind of the expectation-maximization algorithm)
        t0 = datetime.now()

        # x is a collection of sequences with different lengths (its a jagged array of observed sequences)
        V = max(max(xi) for xi in x) + 1
        N = len(x)

        if initial_probs is not None:
            pi, A, B = initial_probs
        else:
            # initialize distributions randomly
            np.random.seed(random_state)
            pi = np.ones(self.M) / self.M
            A = self._random_normalized(self.M, self.M)
            B = self._random_normalized(self.M, V)

        x_seq, cost = self.set(pi, A, B)

        pi_update = self.pi - learning_rate * T.grad(cost, self.pi)
        A_update = self.A - learning_rate * T.grad(cost, self.A)
        B_update = self.B - learning_rate * T.grad(cost, self.B)

        if self.use_softmax is False:
            # because we do not use softmax we have to normalise so that it is probability distribution (it sums to 1)
            pi_update = pi_update / pi_update.sum()
            A_update = A_update / A_update.sum(axis=1).dimshuffle(0, 'x')
            B_update = B_update / B_update.sum(axis=1).dimshuffle(0, 'x')

        updates = [
            (self.pi, pi_update),
            (self.A, A_update),
            (self.B, B_update),
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
        return history, (self.pi.get_value(), self.A.get_value(), self.B.get_value())

    def _random_normalized(self, d1, d2):
        x = np.random.random((d1, d2))
        return x / x.sum(axis=1, keepdims=True)

    def set(self, pi, A, B):
        self.pi = theano.shared(pi)
        self.A = theano.shared(A)
        self.B = theano.shared(B)

        if self.use_softmax is True:
            pi_distr = T.nnet.softmax(self.pi).flatten()  # softmax returns 1xD if input is a 1-D array of size D
            A_distr = T.nnet.softmax(self.A)
            B_distr = T.nnet.softmax(self.B)
        else:
            pi_distr = self.pi
            A_distr = self.A
            B_distr = self.B

        def cals_prob_step(t, prev_alpha, x_seq):
            alpha = B_distr[:, x_seq[t]] * prev_alpha.dot(A_distr)
            s = alpha.sum()
            return alpha / s, s

        x_seq = T.ivector('x_seq')
        [alpha, scale], _ = theano.scan(
            fn=cals_prob_step,
            sequences=T.arange(1, x_seq.shape[0]),
            outputs_info=[pi_distr * B_distr[:, x_seq[0]], None],
            n_steps=x_seq.shape[0] - 1,
            non_sequences=x_seq
        )

        cost = -T.log(scale).sum()
        self.cost_op = theano.function(
            inputs=[x_seq],
            outputs=cost,
            allow_input_downcast=True
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
