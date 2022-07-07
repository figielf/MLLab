import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class hidden_layer:
    def __init__(self, D, M, init_params=None):
        self.D = D  # imput_dim
        self.M = M  # output_dim

        if init_params is None:
            W_init = (np.random.randn(self.D, self.M) / np.sqrt(self.D)).astype(np.float32)
            b_init = np.zeros(self.M).astype(np.float32)
        else:
            W_init, b_init = init_params

        self.W = tf.Variable(W_init)
        self.b = tf.Variable(b_init)

    def forward(self, x):
        return tf.nn.sigmoid(tf.matmul(x, self.W) + self.b)


class ann_classifier:
    def __init__(self, sizes_of_hidden_layers):
        self.sizes_of_hidden_layers = sizes_of_hidden_layers
        self.K = None
        self.hidden_layers = None
        self._session = None

    def _init_params(self, init_params=None):
        D = self.sizes_of_hidden_layers[-1]
        if init_params is None:
            W_init = (np.random.randn(D, self.K) / np.sqrt(D)).astype(np.float32)
            b_init = np.zeros(self.K).astype(np.float32)
        else:
            W_init, b_init = init_params

        self.W = tf.Variable(W_init)
        self.b = tf.Variable(b_init)

    def set_session(self, session):
        self._session = session

    def fit(self, X, Y, Xtest, Ytest, n_epochs=1, batch_size=100, learning_rate=1e-3, init_params=None):
        N, D = X.shape
        self.D = D
        self.K = len(set(Y))
        n_batches = N // batch_size

        self._init_params(init_params)
        self.hidden_layers = []
        prev_l_size = D
        for l_size in self.sizes_of_hidden_layers:
            self.hidden_layers.append(hidden_layer(prev_l_size, l_size, init_params))
            prev_l_size = l_size

        D = self.hidden_layers[0].D
        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, D))
        y_input = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        y_hat_logits = self.forward_logits(self._x_input)
        self._prediction_op = tf.argmax(input=y_hat_logits, axis=1)

        cost_op = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_input,
                logits=y_hat_logits,
            )
        )

        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)

        print(n_batches)
        log_step = np.max([n_batches//5, 1])
        history = []
        self._session.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                x_batch = X[j * batch_size: (j * batch_size + batch_size)]
                y_batch = Y[j * batch_size: (j * batch_size + batch_size)]
                _, train_cost, train_pred = self._session.run((train_op, cost_op, self._prediction_op),
                                                              feed_dict={self._x_input: x_batch, y_input: y_batch})
                current_params = self._get_current_parm_values()
                train_error = self._error_rate(train_pred, y_batch)

                if Xtest is not None:
                    test_cost, test_pred = self._session.run((cost_op, self._prediction_op),
                                                             feed_dict={self._x_input: Xtest, y_input: Ytest})
                    test_error = self._error_rate(test_pred, Ytest)
                    if j % log_step == 0:
                        print(
                            f'ann, epoch:{i}, batch:{j} - train cost:{train_cost}, test cost:{test_cost}, train error:{train_error}, test_error: {test_error}')
                    history.append((train_cost, test_cost, train_error, test_error, current_params))
                else:
                    if j % log_step == 0:
                        print(
                            f'ann, epoch:{i}, batch:{j} - train cost:{train_cost}, train error:{train_error}')
                    history.append((train_cost, train_error, current_params))

        return history

    def forward_logits(self, x):
        z = x
        for l in self.hidden_layers:
            z = l.forward(z)
        return tf.matmul(z, self.W) + self.b

    def predict(self, x):
        y_hat = self._session.run(self._prediction_op, feed_dict={self._x_input: x})
        return y_hat

    def _error_rate(self, p, t):
        return np.mean(p != t)

    def _get_current_parm_values(self):
        cur_params = {}
        j = 0
        for i, l in enumerate(self.hidden_layers):
            cur_params['W_' + str(i)] = l.W.eval()
            cur_params['b_' + str(i)] = l.b.eval()
            j = i
        cur_params['W_' + str(j+1)] = self.W.eval()
        cur_params['b_' + str(j+1)] = self.b.eval()
        return cur_params
