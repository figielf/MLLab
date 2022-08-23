from logging import exception

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from activations import sigmoid


if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class word2vec_tf:
    def __init__(self, D, word2idx, context_size=2, method='skipgram'):
        self.D = D
        self.word2idx = word2idx
        self.context_size = context_size
        self.V = len(self.word2idx)
        if method != 'skipgram':
            raise exception('Only skipgram method to estimate embedding matrix is supported')
        self.W1 = tf.Variable(np.random.randn(self.V, self.D).astype(np.float32))  # input-to-hidden
        self.W2 = tf.Variable(np.random.randn(self.D, self.V).T.astype(np.float32))  # hidden-to-output

    def fit(self, x, word2idx, n_epochs=20, drop_threshold=1e-5):
        tokens_distribution = self._build_negative_sampling_distribution(x, len(word2idx))

        p_drop = 1 - np.sqrt(drop_threshold / tokens_distribution)

        _x_centre = tf.compat.v1.placeholder(tf.int32, shape=(None))
        _x_neg_centre = tf.compat.v1.placeholder(tf.int32, shape=(None))
        _x_context = tf.compat.v1.placeholder(tf.int32, shape=(None))
        _target = tf.compat.v1.placeholder(tf.int32, shape=(None))

        pos_h = tf.nn.embedding_lookup(params=self.W1, ids=_x_centre)  # h = self.W1[centre]
        context = tf.nn.embedding_lookup(params=self.W2, ids=_x_context)
        pos_logits = self._dot(pos_h, context)  # logits = h.dot(self.W2[:, context])
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones(tf.shape(input=pos_logits)),
            logits=pos_logits
        )

        neg_h = tf.nn.embedding_lookup(params=self.W1, ids=_x_neg_centre)
        neg_logits = self._dot(neg_h, context)
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros(tf.shape(input=neg_logits)),
            logits=neg_logits
        )

        loss = tf.reduce_mean(input_tensor=pos_loss) + tf.reduce_mean(input_tensor=neg_loss)

        train_op = tf.compat.v1.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
        # train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
        with tf.compat.v1.Session() as session:
            init_op = tf.compat.v1.global_variables_initializer()
            session.run(init_op)

            history = []
            for epoch in range(n_epochs):
                x_train = shuffle(x)
                cost = 0
                centres = []
                neg_centres = []
                targets = []
                for i, seq in enumerate(x_train):
                    # dropout
                    seq = [w for w in seq if np.random.random() < (1 - p_drop[w])]
                    if len(seq) < 2:
                        continue

                    # randomly order words so we don't always see samples in the same order
                    randomly_ordered_centre_idxs = np.random.choice(len(seq), size=len(seq), replace=False)
                    for centre_idx in randomly_ordered_centre_idxs:
                        centre, context = self._get_window(centre_idx, seq)

                        neg_centre = self._sample_negative_centre(tokens_distribution)

                        n = len(context)
                        centres += [centre] * n
                        neg_centres += [neg_centre] * n
                        targets += context

                    if len(centres) >= 128:
                        _, c = session.run(
                            (train_op, loss),
                            feed_dict={
                                _x_centre: centres,
                                _x_neg_centre: neg_centres,
                                _x_context: targets,
                            }
                        )
                        cost += c

                        centres = []
                        neg_centres = []
                        targets = []

                print(f'epoch {epoch} - loss: {cost}')
                history.append(cost)

                W, VT = session.run((self.W1, self.W2))
        return W, VT.T, history

    def _dot(self, A, B):
        C = A * B
        return tf.reduce_sum(input_tensor=C, axis=1)

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

    def _forward_logits(self, centre, context):
        h = self.W1[centre]
        logits = h.dot(self.W2[:, context])
        return logits

