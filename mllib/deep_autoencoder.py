import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class autoencoder_layer:
    def __init__(self, in_size, out_size, init_weights=None):
        if init_weights is None:
            #W_init = (np.random.randn(D, self.M) / np.sqrt(D)).astype(np.float32)
            W_init = (np.random.randn(in_size, out_size)).astype(np.float32)
            b_in_init = np.zeros(out_size).astype(np.float32)
            b_out_init = np.zeros(in_size).astype(np.float32)  # this bias will be used in second layer use (mirroring flow)
        else:
            W_init, b_in_init, b_out_init = init_weights

        self.W = tf.Variable(W_init)
        self.b_in = tf.Variable(b_in_init)
        self.b_out = tf.Variable(b_out_init)

    def forward_in(self, x):
        return tf.nn.sigmoid(tf.matmul(x, self.W) + self.b_in)

    def forward_out(self, x):
        return tf.nn.sigmoid(self.forward_out_logits(x))

    def forward_out_logits(self, x):
        return tf.matmul(x, tf.transpose(self.W)) + self.b_out


class deep_autoencoder:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self._session = None
        self.layers = None
        self.D = None

    def _prepare_operations(self):
        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.D))
        self._x_hat = self.forward_output(self._x_input)
        self._x_centre = self.forward_in(self._x_input)

        x_hat_logits = self.forward_logits(self._x_input)
        self._cost = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._x_input,
                logits=x_hat_logits,
            )
        )

        self._train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(self._cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(self.cost)

    def set_session(self, session):
        self._session = session

    def fit(self, X, n_epochs=10, batch_size=100):
        N, D = X.shape
        self.D = D
        n_batches = N // batch_size

        if self.layers is None:
            self.layers = []
            in_size = D
            for layer_size in self.hidden_layer_sizes:
                self.layers.append(autoencoder_layer(in_size, layer_size))
                in_size = layer_size

        self._prepare_operations()

        history = []
        self._session.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_epochs):
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_size : (j * batch_size + batch_size)]
                _, c = self._session.run((self._train_op, self._cost), feed_dict={self._x_input: batch})
                if j % 100 == 0:
                    print(f'deep_autoencoder - cost after epoch:{i}, batch:{j} - {c}')
                history.append(c)
        return history

    def forward_in(self, x):
        x_in = x
        for l in self.layers:
            l_out = l.forward_in(x_in)
            x_in = l_out
        return l_out

    def forward_out_logits(self, x_centre):
        n_layers = len(self.layers)
        x_in = x_centre
        for l in range(n_layers - 1, 0, -1):
            l_out = self.layers[l].forward_out(x_in)
            x_in = l_out
        return self.layers[0].forward_out_logits(x_in)

    def forward_logits(self, x):
        x_centre = self.forward_in(x)
        return self.forward_out_logits(x_centre)

    def forward_output(self, x):
        return tf.nn.sigmoid(self.forward_logits(x))

    def map2center(self, x):
        _, D = x.shape
        assert self.D == D
        return self._session.run(self._x_centre, feed_dict={self._x_input: x})

    def predict(self, X):
        _, D = X.shape
        assert self.D == D
        return self._session.run(self._x_hat, feed_dict={self._x_input: X})

