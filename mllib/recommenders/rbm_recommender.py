import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class rbm_recommender:
    # restricted boltzmann machine implementation with bernoulli hidden layer and K-class visible layer which
    #   this can be applied to recommender systems as it nicely handles missing ratings
    #   because K-class one hot encoded dimention has all zeros is such case

    def __init__(self, n_items, n_rating_classes, hidden_layer_size, init_weights=None):
        self.D = n_items
        self.K = n_rating_classes
        self.M = hidden_layer_size  # hidden vector dim
        self.W = None
        self.c = None
        self.b = None
        self._initialize_variables(init_weights)
        self._prepare_operations()
        self._session = None

    def _initialize_variables(self, init_weights):
        if init_weights is None:
            W_init = (np.random.randn(self.D, self.K, self.M) * np.sqrt(2.0 / self.M)).astype(np.float32)
            #W_init = (np.random.randn(self.D, self.K, self.M)).astype(np.float32)
            c_init = np.zeros(self.M).astype(np.float32)
            b_init = np.zeros((self.D, self.K)).astype(np.float32)
        else:
            W_init, c_init, b_init = init_weights

        self.W = tf.Variable(W_init)
        self.c = tf.Variable(c_init)
        self.b = tf.Variable(b_init)

    def _prepare_operations(self):
        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.D))
        x = tf.cast(self._x_input * 2 - 1, tf.int32)  # cast to int 0-9
        x = tf.one_hot(x, self.K)  # x shape=(N, self.D, self.K)

        h = self.prob_h_eq_1_given_v(x)
        h_random_sample = self.bernoulli_sample(h)
        logits_v = self.logits_v_eq_1_given_h(h_random_sample)
        categorical_distrib = tf.compat.v1.distributions.Categorical(logits_v)
        v_prim_random_sample = categorical_distrib.sample()
        v_prim_random_sample = tf.one_hot(v_prim_random_sample, depth=self.K)  # change to shape=(N, D, K)

        # mask X_sample to remove missing ratings
        mask2d = tf.cast(self._x_input > 0, tf.float32)
        mask3d = tf.stack([mask2d] * self.K, axis=-1)  # repeat K times in last dimension
        v_prim_random_sample_masked = v_prim_random_sample * mask3d

        print('x:', x)
        print('v_prim_random_sample_masked:', v_prim_random_sample_masked)
        L = tf.reduce_mean(self.free_energy(x)) - tf.reduce_mean(self.free_energy(v_prim_random_sample_masked))
        self._train_op = tf.compat.v1.train.AdamOptimizer(1e-2).minimize(L)

        x_hat_logits = self.forward_logits(x)
        self._cost = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(x),
                logits=x_hat_logits
            ))

        rating_classes = tf.constant((np.arange(10) + 1).astype(np.float32) / 2)
        self._x_hat = tf.tensordot(self.forward_output(x), rating_classes, axes=[[2], [0]])

        error = self._x_input - self._x_hat
        self._sse = tf.reduce_sum(mask2d * error * error)

        self._x_input_test = tf.compat.v1.placeholder(tf.float32, shape=(None, self.D))
        error_test = self._x_input_test - self._x_hat
        mask2d_test = tf.cast(self._x_input_test > 0, tf.float32)
        self._sse_test = tf.reduce_sum(mask2d_test * error_test * error_test)

    def set_session(self, session):
        self._session = session

    def fit(self, X, X_test, n_epochs=30, batch_size=100):
        N, D = X.shape  # N - number of users, D - number of items
        assert self.D == D
        n_batches = N // batch_size  # batches of users

        history = {
            'MSE_train': [],
            'MSE_test': []}
        if self._session is None:
            self._session = tf.compat.v1.Session()

        self._session.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_epochs):
            print(f'epoch:{i}')
            X, X_test = shuffle(X, X_test)
            for j in range(n_batches):
                batch = X[j * batch_size: (j * batch_size + batch_size)].toarray()
                _, c = self._session.run((self._train_op, self._cost), feed_dict={self._x_input: batch})
                if j % 100 == 0:
                    print(f'batch:{j} - cost: {c}')

            # calculate SSE after whole epoch updates per batch so that we prevent from out of memory issues
            sse = 0
            sse_test = 0
            n_non_zero_ratings = 0
            n_non_zero_ratings_test = 0
            for j in range(n_batches):
                batch = X[j * batch_size: (j * batch_size + batch_size)].toarray()
                batch_test = X_test[j * batch_size: (j * batch_size + batch_size)].toarray()

                n_non_zero_ratings += np.count_nonzero(batch)
                n_non_zero_ratings_test += np.count_nonzero(batch_test)

                batch_see, batch_test_see = self._session.run(
                    (self._sse, self._sse_test),
                    feed_dict={self._x_input: batch, self._x_input_test: batch_test})

                sse += batch_see
                sse_test += batch_test_see

            mse = sse / n_non_zero_ratings
            mse_test = sse_test / n_non_zero_ratings_test
            print(f'after epoch:{i} - train MSE: {mse}, test MSE: {mse_test}')

            history['MSE_train'].append(mse)
            history['MSE_test'].append(mse_test)
        return history

    def prob_h_eq_1_given_v(self, v):
        _, D, K = v.shape
        assert self.D == D
        assert self.K == K
        return tf.nn.sigmoid(self._dot_2d(v, self.W) + self.c)

    def logits_v_eq_1_given_h(self, h):
        _, M = h.shape
        assert self.M == M
        logits = self._dot_1d(h, self.W) + self.b
        return logits

    def forward_hidden(self, x):
        return self.prob_h_eq_1_given_v(x)

    def forward_logits(self, x):
        z = self.forward_hidden(x)
        return self._dot_1d(z, self.W) + self.b

    def forward_output(self, x):
        return tf.nn.softmax(self.forward_logits(x))

    def predict(self, X):
        _, D, K = X.shape
        assert self.D == D
        assert self.K == K
        return self._session.run(self._x_hat, feed_dict={self._x_input: X})

    def bernoulli_sample(self, p):
        u = tf.random.uniform(tf.shape(p))
        return tf.cast(u < p, dtype=tf.float32)

    def free_energy(self, v):
        term_1 = tf.reduce_sum(self._dot_2d(v, self.b))

        # calculate: log(1 + tf.exp(tf.matmul(v, self.W) + self.c)),
        term_2 = -tf.reduce_sum(input_tensor=tf.nn.softplus(self._dot_2d(v, self.W) + self.c), axis=1)
        return term_1 + term_2

    def _dot_2d(self, a, b):
        # a shape=(N, self.M, self.K)
        # b shape=(self.D, self.K, self.M)
        # returns shape=(N, self.M)
        return tf.tensordot(a, b, axes=[[1, 2], [0, 1]])

    def _dot_1d(self, a, b):
        # a shape=(N, self.M)
        # b shape=(self.D, self.K, self.M)
        # returns shape=(N, self.D, self.K)
        return tf.tensordot(a, b, axes=[[1], [2]])
