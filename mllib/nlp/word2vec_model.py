from logging import exception

import numpy as np
from sklearn.utils import shuffle

from activations import sigmoid


class word2vec:
    def __init__(self, D, word2idx, context_size=2, method='skipgram'):
        self.D = D
        self.word2idx = word2idx
        self.context_size = context_size
        self.V = len(self.word2idx)
        if method != 'skipgram':
            raise exception('Only skipgram method to estimate embedding matrix is supported')
        self.W1 = np.random.randn(self.V, self.D)  # input-to-hidden
        self.W2 = np.random.randn(self.D, self.V)  # hidden-to-output

    def fit(self, x, word2idx, learning_rate=0.025, learning_rate_decay=0, n_epochs=20, drop_threshold=1e-5):
        lr = learning_rate

        tokens_distribution = self._build_negative_sampling_distribution(x, len(word2idx))

        p_drop = 1 - np.sqrt(drop_threshold / tokens_distribution)

        history = []
        for epoch in range(n_epochs):
            x_train = shuffle(x)
            counter = 0
            cumulative_cost = 0
            cost_sample_count = 0
            for i, seq in enumerate(x_train):
                # dropout
                seq = [w for w in seq if np.random.random() < (1 - p_drop[w])]
                if len(seq) < 2:
                    continue

                # randomly order words so we don't always see samples in the same order
                randomly_ordered_centre_idxs = np.random.choice(len(seq), size=len(seq), replace=False)
                for centre_idx in randomly_ordered_centre_idxs:
                    centre, context = self._get_window(centre_idx, seq)

                    # positive sample
                    target = 1
                    c = self._sgd(centre, context, target, lr)
                    cumulative_cost += c
                    cost_sample_count += len(context)

                    # negative sample
                    neg_centre = self._sample_negative_centre(tokens_distribution)
                    target = 0
                    c = self._sgd(neg_centre, context, target, lr)
                    cumulative_cost += c
                    cost_sample_count += len(context)

            cost = cumulative_cost / cost_sample_count
            print(f'epoch {epoch} - loss: {cost}')
            history.append(cost)
            lr = lr - learning_rate_decay
        return self.W1, self.W2, history

    def _sgd(self, centre, context, target, learning_rate):
        h = self.W1[centre]
        prob = sigmoid(h.dot(self.W2[:, context]))

        error = prob - target
        dW2 = np.outer(h, error)
        dW1 = np.sum(error * self.W2[:, context], axis=1)
        for ic, c in enumerate(context):
            self.W2[:, c] = self.W2[:, c] - learning_rate * dW2[:, ic]
        self.W1[centre] = self.W1[centre] - learning_rate * dW1

        eps = 1e-10
        cost = target * np.log(prob + eps) + (1 - target) * np.log(1 - prob + eps)
        return -cost.sum()

    def _get_window(self, centre_idx, sequence):
        centre = sequence[centre_idx]
        left = sequence[:centre_idx][-self.context_size:]
        right = sequence[centre_idx + 1:][:self.context_size]
        context = left + right
        return centre, context

    def _sample_negative_centre(self, p):
        return np.random.choice(len(p), p=p)

    def _build_negative_sampling_distribution(self, x, vocab_size):
        token_counts = np.zeros(vocab_size)
        for seq in x:
            for t in seq:
                token_counts[t] += 1

        token_counts = token_counts ** 0.75  # smoothing
        assert(np.all(token_counts > 0))
        return token_counts / token_counts.sum()
