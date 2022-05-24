import numpy as np


class SparseSecondOrderMarkovModel:
    def __init__(self, n_states, seq_end=None):
        self.n_states = n_states
        self.pi = None
        self.A1 = None
        self.A2 = None
        self.seq_end = seq_end

    def fit(self, x):
        assert len(x) > 0

        # list all states from x at every sequence position t
        x_pi = []  # list(state_0)
        x_A1 = {}  # dict(state_0, list(state_1))
        x_A2 = {}  # dict((state_t_2, state_t_1), list(state_t))
        for idx, x_cur in enumerate(x):
            state_0 = x_cur[0]
            x_pi.append(state_0)
            x_A1[state_0] = x_A1.get(state_0, [])
            if len(x_cur) == 1:
                if self.seq_end is not None:
                    x_A1[state_0].append(self.seq_end)
            else:
                state_1 = x_cur[1]
                x_A1[state_0].append(state_1)
                for t in range(2, len(x_cur)):
                    pair_t_1 = (x_cur[t - 2], x_cur[t - 1])
                    x_A2[pair_t_1] = x_A2.get(pair_t_1, [])
                    x_A2[pair_t_1].append(x_cur[t])

        # count the probability of each transition from dictionaries above,
        # this will give us sparse matrix (implemented as dictionary) of transition states
        self.pi = self._calc_discrete_distribution(x_pi)  # dict(state_0, state_0 probability at first position)

        self.A1 = {}  # dict(state_0, dict(state_1, state_0 -> state_1 transition probability at first two positions position))
        for state_1, cur_state in x_A1.items():
            self.A1[state_1] = self._calc_discrete_distribution(cur_state)

        self.A2 = {}  # dict((state_t_2, state_t_1), dict(state_t, (state_t_2, state_t_1) -> state_t  transition probability in the whole sequence))
        for state_1, cur_state in x_A2.items():
            self.A2[state_1] = self._calc_discrete_distribution(cur_state)

    @staticmethod
    def _calc_discrete_distribution(tokens: list):
        n = len(tokens)
        pdf = {}
        for i, token in enumerate(tokens):
            pdf[token] = pdf.get(token, 0) + 1

        for k, val in pdf.items():
            pdf[k] = val / n
        return pdf


class MAPDiscreteSequenceGenerator:  # Maximum Posteriori
    def __init__(self, likelihood_model):
        self.model = likelihood_model

    def generate(self, max_steps):
        assert max_steps > 0
        x_generated = []

        # generate token 0
        token_0_distribution = self.model.pi
        x_generated.append(self._cdf_inv(token_0_distribution))

        # generate token 1
        if max_steps > 1:
            token_1_distribution = self.model.A1[x_generated[0]]
            next_token = self._cdf_inv(token_1_distribution)
            x_generated.append(self._cdf_inv(token_1_distribution))
            if self.model.seq_end is not None and next_token == self.model.seq_end:
                return x_generated

        # generate tokens >= 2
        for t in range(2, max_steps):
            token_t_distribution = self.model.A2[(x_generated[-2], x_generated[-1])]
            next_token = self._cdf_inv(token_t_distribution)
            x_generated.append(next_token)
            if self.model.seq_end is not None and next_token == self.model.seq_end:
                break

        return x_generated

    @staticmethod
    def _cdf_inv(discrete_distribution: dict):
        u = np.random.random()
        cdf = 0.0
        for token, prob in discrete_distribution.items():
            cdf += prob
            if u < cdf:
                return token
        raise Exception('Unexpected line execution. Probably provided discrete distribution was not correct')
