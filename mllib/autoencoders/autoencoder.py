import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class autoencoder:
    def __init__(self, D, M, id=0, init_weights=None):
        self.D = D
        self.M = M
        self.id = str(id)
        self.W = None
        self.bh = None
        self.bo = None
        self._session = None
        self._initialize_variables(init_weights)
        self._prepare_operations()

    def _initialize_variables(self, init_weights):
        if init_weights is None:
            #W_init = (np.random.randn(D, self.M) / np.sqrt(D)).astype(np.float32)
            W_init = (np.random.randn(self.D, self.M)).astype(np.float32)
            bh_init = np.zeros(self.M).astype(np.float32)
            bo_init = np.zeros(self.D).astype(np.float32)
        else:
            W_init, bh_init, bo_init = init_weights

        self.W = tf.Variable(W_init)
        self.bh = tf.Variable(bh_init)
        self.bo = tf.Variable(bo_init)

    def _prepare_operations(self):
        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.D))
        self._z = self.forward_hidden(self._x_input) # for transform() later
        self._x_hat = self.forward_output(self._x_input)

        x_hat_logits = self.forward_logits(self._x_input)
        self._cost = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._x_input,
                logits=x_hat_logits,
            )
        )

        self._train_op = tf.compat.v1.train.AdamOptimizer(1e-1).minimize(self._cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(self.cost)

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
                    print(f'autoencoder:{self.id} - cost after epoch:{i}, batch:{j} - {c}')
                history.append(c)
        return history

    def forward_hidden(self, x):
        _, D = x.shape
        assert self.D == D
        return tf.nn.sigmoid(tf.matmul(x, self.W) + self.bh)

    def forward_logits(self, x):
        z = self.forward_hidden(x)
        return tf.matmul(z, tf.transpose(a=self.W)) + self.bo

    def forward_output(self, x):
        return tf.nn.sigmoid(self.forward_logits(x))

    def transform(self, X):
        _, D = X.shape
        assert self.D == D
        return self._session.run(self._z, feed_dict={self._x_input: X})

    def predict(self, X):
        _, D = X.shape
        assert self.D == D
        return self._session.run(self._x_hat, feed_dict={self._x_input: X})

