import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class rbm:
    # bernoulli restricted boltzmann machine implementation

    def __init__(self, D, M, id=0, init_weights=None):
        self.D = D
        self.M = M
        self.id = str(id)
        self.W = None
        self.c = None
        self.b = None
        self._session = None
        self._initialize_variables(init_weights)
        self._prepare_operations()

    def _initialize_variables(self, init_weights):
        if init_weights is None:
            W_init = (np.random.randn(self.D, self.M) * np.sqrt(2.0 / self.M)).astype(np.float32)
            #W_init = (np.random.randn(self.D, self.M)).astype(np.float32)
            c_init = np.zeros(self.M).astype(np.float32)
            b_init = np.zeros(self.D).astype(np.float32)
        else:
            W_init, c_init, b_init = init_weights

        self.W = tf.Variable(W_init)
        self.c = tf.Variable(c_init)
        self.b = tf.Variable(b_init)

    def _prepare_operations(self):
        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.D))
        self._h = self.prob_h_eq_1_given_v(self._x_input)
        h_random_sample = self.bernoulli_sample(self._h)
        self._v = self.prob_v_eq_1_given_h(h_random_sample)
        v_prim_random_sample = self.bernoulli_sample(self._v)

        L = tf.reduce_mean(self.free_energy(self._x_input)) - tf.reduce_mean(self.free_energy(v_prim_random_sample))
        self._train_op = tf.compat.v1.train.AdamOptimizer(1e-2).minimize(L)

        x_hat_logits = self.forward_logits(self._x_input)
        self._cost = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._x_input,
                logits=x_hat_logits
            ))

    def set_session(self, session):
        self._session = session

    def fit(self, X, n_epochs=1, batch_size=100):
        N, D = X.shape
        assert self.D == D
        n_batches = N // batch_size

        history = []
        self._session.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_epochs):
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_size : (j * batch_size + batch_size)]
                _, c = self._session.run((self._train_op, self._cost), feed_dict={self._x_input: batch})
                if j % 100 == 0:
                    print(f'rmb:{self.id} - cost after epoch:{i}, batch:{j} - {c}')
                history.append(c)
        return history

    def prob_h_eq_1_given_v(self, v):
        _, D = v.shape
        assert self.D == D
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.c)

    def prob_v_eq_1_given_h(self, h):
        _, M = h.shape
        assert self.M == M
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.b)

    def forward_hidden(self, x):
        return self.prob_h_eq_1_given_v(x)

    def forward_logits(self, x):
        z = self.forward_hidden(x)
        return tf.matmul(z, tf.transpose(self.W)) + self.b

    def forward_output(self, x):
        return tf.nn.sigmoid(self.forward_logits(x))

    def transform(self, X):
        _, D = X.shape
        assert self.D == D
        return self._session.run(self._h, feed_dict={self._x_input: X})

    def predict(self, X):
        _, D = X.shape
        assert self.D == D
        return self._session.run(self._x_hat, feed_dict={self._x_input: X})

    def bernoulli_sample(self, p):
        u = tf.random.uniform(tf.shape(p))
        return tf.cast(u < p, dtype=tf.float32)

    def free_energy(self, v):
        term_1 = -tf.matmul(v, tf.reshape(self.b, (self.D, 1)))

        # calculate: log(1 + tf.exp(tf.matmul(v, self.W) + self.c)),
        term_2 = -tf.reduce_sum(
            input_tensor=tf.nn.softplus(tf.matmul(v, self.W) + self.c),
            axis=1
        )

        return tf.reshape(term_1, (-1,)) + term_2

