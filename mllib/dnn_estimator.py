import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class dnn:
    def __init__(self, D, hidden_layer_output_sizes, K, base_model, final_layer_init_weights=None):
        self.D = D
        self.hidden_layer_output_sizes = hidden_layer_output_sizes
        self.K = K
        self.base_model = base_model
        self.W = None
        self.b = None
        self.hidden_layers = None
        self._initialize_variables(final_layer_init_weights, hidden_layer_output_sizes[-1])
        self._prepare_operations()
        self._session = None

    def _initialize_variables(self, init_weights, final_layer_D):
        if init_weights is None:
            # W_init = (np.random.randn(self.D, self.K) / np.sqrt(self.D)).astype(np.float32)
            W_init = (np.random.randn(final_layer_D, self.K)).astype(np.float32)
            b_init = np.zeros(self.K).astype(np.float32)
        else:
            W_init, b_init = init_weights

        self.W = tf.Variable(W_init)
        self.b = tf.Variable(b_init)

        print(f'dmm model construction with {len(self.hidden_layer_output_sizes)} hidden layers')
        self.hidden_layers = []
        layer_input_size = self.D
        for i, hidden_units in enumerate(self.hidden_layer_output_sizes):
            print(f'layer {i} - with {hidden_units} hidden units')
            hidden_layer_unsupervised_model = self.base_model(layer_input_size, hidden_units, id=i)
            self.hidden_layers.append(hidden_layer_unsupervised_model)
            layer_input_size = hidden_units

    def _prepare_operations(self):
        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.D))
        self._y_input = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        y_hat_logits = self.forward_logits(self._x_input)
        self._cost = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_hat_logits,
                labels=self._y_input,
            )
        )
        self._train_op = tf.compat.v1.train.AdamOptimizer(1e-2).minimize(self._cost)
        self._prediction = tf.argmax(input=y_hat_logits, axis=1)

    def set_session(self, session):
        self._session = session
        for layer in self.hidden_layers:
            layer.set_session(self._session)

    def fit(self, X, Y, Xtest, Ytest, pretrain=True, n_epochs=1, batch_size=100):
        N, D = X.shape
        assert D == self.D
        n_batches = N // batch_size

        if pretrain:
            print('will use pretraining...')
            pretrain_epochs = 1
        else:
            print('will not use pretraining...')
            pretrain_epochs = 0

        layer_input = X
        for layer in self.hidden_layers:
            layer.fit(layer_input, n_epochs=pretrain_epochs)
            prev_layer_output = layer.transform(layer_input)
            layer_input = prev_layer_output

        history = []
        print(f'Starting supervised training of dnn model ...')
        self._session.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_epochs):
            print(f'supervised dnn, epoch:{i}')
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                x_batch = X[j * batch_size: (j * batch_size + batch_size)]
                y_batch = Y[j * batch_size: (j * batch_size + batch_size)]

                _, train_cost, train_pred = self._session.run((self._train_op, self._cost, self._prediction),
                                                              feed_dict={self._x_input: x_batch,
                                                                         self._y_input: y_batch})

                test_cost, test_pred = self._session.run(
                    (self._cost, self._prediction),
                    feed_dict={self._x_input: Xtest, self._y_input: Ytest})

                train_error = self._error_rate(train_pred, y_batch)
                test_error = self._error_rate(test_pred, Ytest)
                if j % 100 == 0:
                    print(
                       f'supervised dnn, epoch:{i}, batch:{j} - train cost:{train_cost}, test cost:{test_cost}, train error:{train_error}, test_error: {test_error}')
                history.append([train_cost, test_cost, train_error, test_error])
        return np.array(history)

    def forward_logits(self, x):
        _, D = x.shape
        assert D == self.D
        current_input = x
        for ae in self.hidden_layers:
            z = ae.forward_hidden(current_input)
            current_input = z

        logits = tf.matmul(current_input, self.W) + self.b
        return logits

    def predict(self, X):
        _, D = X.shape
        assert D == self.D
        return self._session.run(self._prediction, feed_dict={self._x_input: X})

    def _error_rate(self, p, t):
        return np.mean(p != t)
