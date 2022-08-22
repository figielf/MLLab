import numpy as np
import tensorflow as tf

from nlp.processing.term_term_matrix import term_term_matrix

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class glove:
    def __init__(self, D, V, context_size, start_end_tokens):
        self.D = D
        self.V = V
        self.context_size = context_size
        self.start_end_tokens = start_end_tokens
        self.W = np.random.randn(self.V, self.D) / np.sqrt(self.V + self.D)
        self.b = np.zeros(self.V)
        self.U = np.random.randn(self.V, self.D) / np.sqrt(self.V + self.D)
        self.c = np.zeros(self.V)

    def fit(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, n_epochs=10):
        # fit weights by gradient descent of factorized term-term matrix

        logX, fX = self._prepare_training_data(sentences, cc_matrix, xmax, alpha,
                                               start_end_tokens_to_add=self.start_end_tokens)
        mu = logX.mean()

        history = []
        for epoch in range(n_epochs):
            cost, delta = self._calc_cost(logX, fX, mu)
            history.append(cost)
            print(f'epoch: {epoch}, cost: {cost}')

            # update self.W
            for i in range(self.V):
                self.W[i] = self.W[i] - learning_rate * ((fX[i, :] * delta[i, :]).dot(self.U) + reg * self.W[i])
            # self.W = self.W - learning_rate * reg * self.W

            # update self.b
            for i in range(self.V):
                self.b[i] = self.b[i] - learning_rate * (fX[i, :].dot(delta[i, :]) + reg * self.b[i])

            # update self.U
            for j in range(self.V):
                self.U[j] = self.U[j] - learning_rate * ((fX[:, j] * delta[:, j]).dot(self.W) + reg * self.U[j])
            # self.U = self.U - learning_rate * reg * self.U

            # update self.c
            for j in range(self.V):
                self.c[j] = self.c[j] - learning_rate * (fX[:, j].dot(delta[:, j]) + reg * self.c[j])

        return history

    def fit_by_tf(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, n_epochs=10):
        # fit weights by alternating least squares of factorized term-term matrix

        tf_W = tf.Variable(self.W.astype(np.float32))
        tf_b = tf.Variable(self.b.reshape(self.V, 1).astype(np.float32))
        tf_U = tf.Variable(self.U.astype(np.float32))
        tf_c = tf.Variable(self.c.reshape(1, self.V).astype(np.float32))

        tf_logX = tf.compat.v1.placeholder(tf.float32, shape=(self.V, self.V))
        tf_fX = tf.compat.v1.placeholder(tf.float32, shape=(self.V, self.V))

        logX, fX = self._prepare_training_data(sentences, cc_matrix, xmax, alpha,
                                               start_end_tokens_to_add=self.start_end_tokens)
        mu = logX.mean()

        error = tf.matmul(tf_W, tf.transpose(tf_U)) + tf_b + tf_c + mu - logX
        cost = tf.reduce_sum(fX * error * error)
        regularized_cost = cost
        for param in (tf_W, tf_b, tf_U, tf_c):
            regularized_cost += reg * tf.reduce_sum(input_tensor=param * param)

        train_op = tf.compat.v1.train.MomentumOptimizer(
            learning_rate,
            momentum=0.9
        ).minimize(regularized_cost)

        with tf.compat.v1.Session() as session:
            session = tf.compat.v1.Session()
            init_op = tf.compat.v1.global_variables_initializer()
            session.run(init_op)

            history = []
            for epoch in range(n_epochs):
                _, c = session.run((train_op, cost), feed_dict={tf_logX: logX, tf_fX: fX}
                        )
                history.append(c)
                print(f'epoch: {epoch}, cost: {c}')

            self.W, self.b, self.U, self.c = session.run([tf_W, tf_b, tf_U, tf_c])

        return history

    def fit_by_als(self, sentences, cc_matrix=None, reg=0.1, xmax=100, alpha=0.75, n_epochs=10):
        # fit weights by alternating least squares of factorized term-term matrix

        logX, fX = self._prepare_training_data(sentences, cc_matrix, xmax, alpha,
                                               start_end_tokens_to_add=self.start_end_tokens)
        mu = logX.mean()

        history = []
        for epoch in range(n_epochs):
            cost, _ = self._calc_cost(logX, fX, mu)
            history.append(cost)
            print(f'epoch: {epoch}, cost: {cost}')

            # update self.W
            for i in range(self.V):
                a = reg * np.eye(self.D) + (fX[i, :] * self.U.T).dot(self.U)
                b = (fX[i, :] * (logX[i, :] - self.b[i] - self.c - mu)).dot(self.U)
                self.W[i] = np.linalg.solve(a, b)

            # update self.b
            for i in range(self.V):
                a = fX[i, :].sum() + reg
                b = fX[i, :].dot(logX[i, :] - self.W[i].dot(self.U.T) - self.c - mu)
                self.b[i] = b / a

            # update self.U
            for j in range(self.V):
                a = reg * np.eye(self.D) + (fX[:, j] * self.W.T).dot(self.W)
                b = (fX[:, j] * (logX[:, j] - self.b - self.c[j] - mu)).dot(self.W)
                self.U[j] = np.linalg.solve(a, b)

            # update self.c
            for j in range(self.V):
                a = fX[:, j].sum() + reg
                b = fX[:, j].dot(logX[:, j] - self.W.dot(self.U[j]) - self.b - mu)
                self.c[i] = b / a

        return history

    def _calc_cost(self, logX, fX, mu):
        prediction = self.W.dot(self.U.T) + self.b.reshape(self.V, 1) + self.c.reshape(1, self.V) + mu
        error = prediction - logX
        cost = np.sum(fX * error * error)
        return cost, error

    def _prepare_training_data(self, sentences, cc_matrix, xmax, alpha, start_end_tokens_to_add):
        if cc_matrix is None:
            print('About to create Term-Term matrix ...')
            cc_matrix = term_term_matrix(sentences, self.V, self.context_size, start_end_tokens_to_add)
            print('Term-Term matrix created')

        X = cc_matrix
        fX = np.ones((self.V, self.V))
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha

        logX = np.log(X + 1)
        return logX, fX
